package minions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"

	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/openai/openai-go/shared"
)

// MinionsOrchestrationService coordinates the MinionS protocol loop.
type MinionsOrchestrationService struct{}

// NewMinionsOrchestrationService constructs the service.
func NewMinionsOrchestrationService() *MinionsOrchestrationService {
	return &MinionsOrchestrationService{}
}

// OrchestrateMinionS runs the MinionS protocol loop (non-streaming).
func (s *MinionsOrchestrationService) OrchestrateMinionS(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	localProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
) (*openai.ChatCompletion, error) {
	const maxRounds = 8
	var (
		roundResults []*openai.ChatCompletion
		round        = 0
	)
	for round < maxRounds {
		// a) Decompose remotely
		instructions, _, err := s.remoteDecompose(remoteProv, req, roundResults)
		if err != nil {
			return nil, err
		}
		// b) Execute & filter locally
		minionInputs := s.distributeInstructions(req.Messages, instructions)
		rawResults := s.executeLocalMinionsParallel(localProv, minionInputs)
		roundResults = s.filterMinionResults(rawResults)
		// c) Aggregate remotely (non-streaming)
		final, needMore, err := s.remoteAggregate(remoteProv, req, roundResults)
		if err != nil {
			return nil, err
		}
		if final != nil {
			return final, nil
		}
		if !needMore {
			break
		}
		round++
	}
	return nil, errors.New("MinionS protocol did not converge")
}

// OrchestrateMinionSStream runs the MinionS protocol loop (streaming).
// Returns a stream of ChatCompletionChunk once aggregation is final.
func (s *MinionsOrchestrationService) OrchestrateMinionSStream(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	localProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	const maxRounds = 8
	var (
		roundResults []*openai.ChatCompletion
		round        = 0
	)
	for round < maxRounds {
		// a) Decompose remotely
		instructions, _, err := s.remoteDecompose(remoteProv, req, roundResults)
		if err != nil {
			return nil, err
		}
		// b) Execute & filter locally
		minionInputs := s.distributeInstructions(req.Messages, instructions)
		rawResults := s.executeLocalMinionsParallel(localProv, minionInputs)
		roundResults = s.filterMinionResults(rawResults)
		// c) Aggregate remotely (streaming + buffered)
		stream, needMore, err := s.remoteAggregateStreamBuffered(remoteProv, req, roundResults)
		if err != nil {
			return nil, err
		}
		if !needMore {
			return stream, nil
		}
		round++
	}
	return nil, errors.New("MinionS streaming protocol did not converge")
}

// --- Protocol Step Implementations ---

func (s *MinionsOrchestrationService) remoteDecompose(
	remoteProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	previous []*openai.ChatCompletion,
) (instructions []string, chunkStrategy string, err error) {
	var prevSummaries []string
	for _, r := range previous {
		for _, ch := range r.Choices {
			prevSummaries = append(prevSummaries, ch.Message.Content)
		}
	}
	systemPrompt := "You are a large language model. Decompose the user query into atomic instructions for a local LM, and specify a chunking/selection strategy for the document. Respond as JSON: {\"instructions\": [\"...\"], \"chunk_strategy\": \"...\"}"
	userPrompt := fmt.Sprintf(
		"Query: %s\nPrevious Results: %s",
		getUserQuery(req),
		strings.Join(prevSummaries, "\n"),
	)
	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(userPrompt),
		},
		Model:          req.Model,
		ResponseFormat: jsonSchemaResponseFormat(),
	}
	resp, err := remoteProv.Chat().Completions().CreateCompletion(&param)
	if err != nil || len(resp.Choices) == 0 {
		return nil, "", errors.New("remote decomposition failed")
	}
	var parsed struct {
		Instructions  []string `json:"instructions"`
		ChunkStrategy string   `json:"chunk_strategy"`
	}
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, "", errors.New("failed to parse decomposition response")
	}
	return parsed.Instructions, parsed.ChunkStrategy, nil
}

func (s *MinionsOrchestrationService) distributeInstructions(
	chunks []openai.ChatCompletionMessageParamUnion,
	instructions []string,
) []struct {
	Chunk       openai.ChatCompletionMessageParamUnion
	Instruction string
} {
	var inputs []struct {
		Chunk       openai.ChatCompletionMessageParamUnion
		Instruction string
	}
	for _, chunk := range chunks {
		for _, instr := range instructions {
			inputs = append(inputs, struct {
				Chunk       openai.ChatCompletionMessageParamUnion
				Instruction string
			}{
				Chunk:       chunk,
				Instruction: instr,
			})
		}
	}
	return inputs
}

func (s *MinionsOrchestrationService) executeLocalMinionsParallel(
	localProv provider_interfaces.LLMProvider,
	inputs []struct {
		Chunk       openai.ChatCompletionMessageParamUnion
		Instruction string
	},
) []*openai.ChatCompletion {
	results := make([]*openai.ChatCompletion, len(inputs))
	var wg sync.WaitGroup
	wg.Add(len(inputs))
	for i, input := range inputs {
		go func(i int, input struct {
			Chunk       openai.ChatCompletionMessageParamUnion
			Instruction string
		},
		) {
			defer wg.Done()
			prompt := buildMinionPrompt(input.Instruction, input.Chunk)
			param := openai.ChatCompletionNewParams{
				Messages: []openai.ChatCompletionMessageParamUnion{
					openai.SystemMessage("You are a helpful assistant. Answer only based on the provided chunk. If you cannot answer, abstain."),
					openai.UserMessage(prompt),
				},
				ResponseFormat: jsonSchemaResponseFormat(),
			}
			resp, err := localProv.Chat().Completions().CreateCompletion(&param)
			if err != nil {
				results[i] = nil
				return
			}
			results[i] = resp
		}(i, input)
	}
	wg.Wait()
	return results
}

func (s *MinionsOrchestrationService) filterMinionResults(
	results []*openai.ChatCompletion,
) []*openai.ChatCompletion {
	var filtered []*openai.ChatCompletion
	for _, r := range results {
		if r == nil {
			continue
		}
		keep := false
		for _, ch := range r.Choices {
			lc := strings.ToLower(ch.Message.Content)
			if !strings.Contains(lc, "abstain") &&
				!strings.Contains(lc, "\"answer\": \"none\"") {
				keep = true
				break
			}
		}
		if keep {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func (s *MinionsOrchestrationService) remoteAggregate(
	remoteProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	results []*openai.ChatCompletion,
) (final *openai.ChatCompletion, needMore bool, err error) {
	aggregationPrompt := buildAggregationPrompt(req, results)
	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("You are a helpful assistant. Aggregate the following minion results and decide if the answer is final. Respond as JSON: {\"final\": true/false, \"answer\": \"...\"}"),
			openai.UserMessage(aggregationPrompt),
		},
		Model:          req.Model,
		ResponseFormat: jsonSchemaAggregateResponseFormat(),
	}
	resp, err := remoteProv.Chat().Completions().CreateCompletion(&param)
	if err != nil || len(resp.Choices) == 0 {
		return nil, false, errors.New("remote aggregation failed")
	}
	var parsed struct {
		Final  bool   `json:"final"`
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, false, errors.New("failed to parse aggregation response")
	}
	finalResp := &openai.ChatCompletion{
		Choices: []openai.ChatCompletionChoice{
			{
				Message: openai.ChatCompletionMessage{
					Role:    "assistant",
					Content: parsed.Answer,
				},
			},
		},
	}
	return finalResp, !parsed.Final, nil
}

// --- Streaming aggregation with buffered decode ---

type memoryChunkDecoder struct {
	chunks []*openai.ChatCompletionChunk
	idx    int
	evt    ssestream.Event
}

func newMemoryChunkDecoder(chunks []*openai.ChatCompletionChunk) *memoryChunkDecoder {
	return &memoryChunkDecoder{chunks: chunks}
}

func (d *memoryChunkDecoder) Next() bool {
	if d.idx >= len(d.chunks) {
		return false
	}
	chunk := d.chunks[d.idx]
	d.idx++
	data, err := json.Marshal(chunk)
	if err != nil {
		return false
	}
	d.evt = ssestream.Event{Type: "", Data: data}
	return true
}

func (d *memoryChunkDecoder) Event() ssestream.Event { return d.evt }
func (d *memoryChunkDecoder) Close() error           { return nil }
func (d *memoryChunkDecoder) Err() error             { return nil }

// remoteAggregateStreamBuffered buffers the streaming aggregation.
func (s *MinionsOrchestrationService) remoteAggregateStreamBuffered(
	remoteProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	results []*openai.ChatCompletion,
) (*ssestream.Stream[openai.ChatCompletionChunk], bool, error) {
	aggregationPrompt := buildAggregationPrompt(req, results)
	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("You are a helpful assistant. Aggregate the following minion results and decide if the answer is final. Respond as JSON: {\"final\": true/false, \"answer\": \"...\"}"),
			openai.UserMessage(aggregationPrompt),
		},
		Model:          req.Model,
		ResponseFormat: jsonSchemaAggregateResponseFormat(),
	}
	origStream, err := remoteProv.Chat().Completions().StreamCompletion(&param)
	if err != nil {
		return nil, false, errors.New("remote aggregation streaming failed")
	}
	var (
		chunks         []*openai.ChatCompletionChunk
		contentBuilder strings.Builder
	)
	for origStream.Next() {
		chunk := origStream.Current()
		b, err := json.Marshal(chunk)
		if err != nil {
			continue
		}
		var parsedChunk openai.ChatCompletionChunk
		if err := json.Unmarshal(b, &parsedChunk); err == nil {
			chunks = append(chunks, &parsedChunk)
			if len(parsedChunk.Choices) > 0 {
				contentBuilder.WriteString(parsedChunk.Choices[0].Delta.Content)
			}
		}
	}
	if err := origStream.Err(); err != nil {
		return nil, false, err
	}
	var parsed struct {
		Final  bool   `json:"final"`
		Answer string `json:"answer"`
	}
	if err := json.Unmarshal([]byte(contentBuilder.String()), &parsed); err != nil {
		return nil, false, errors.New("failed to parse aggregation response")
	}
	memDecoder := newMemoryChunkDecoder(chunks)
	stream := ssestream.NewStream[openai.ChatCompletionChunk](memDecoder, nil)
	return stream, !parsed.Final, nil
}

// --- Helpers ---

func buildMinionPrompt(
	instruction string,
	chunk openai.ChatCompletionMessageParamUnion,
) string {
	return fmt.Sprintf(
		"Instruction: %s\n\nChunk:\n%v\n\nReturn your answer as JSON: {\"answer\": \"...\", \"citation\": \"...\", \"explanation\": \"...\"} or abstain if not found.",
		instruction, chunk,
	)
}

func buildAggregationPrompt(
	req *models.ChatCompletionRequest,
	results []*openai.ChatCompletion,
) string {
	var sb strings.Builder
	sb.WriteString("User Query: ")
	sb.WriteString(getUserQuery(req))
	sb.WriteString("\nMinion Results:\n")
	for i, r := range results {
		for _, ch := range r.Choices {
			sb.WriteString(fmt.Sprintf("Minion %d: %s\n", i+1, ch.Message.Content))
		}
	}
	return sb.String()
}

func getUserQuery(req *models.ChatCompletionRequest) string {
	for _, msg := range req.Messages {
		if msg.OfUser != nil {
			return msg.OfUser.Content.OfString.Value
		}
	}
	return ""
}

func jsonSchemaResponseFormat() openai.ChatCompletionNewParamsResponseFormatUnion {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"answer":      map[string]any{"type": "string"},
			"citation":    map[string]any{"type": "string"},
			"explanation": map[string]any{"type": "string"},
		},
		"required": []string{"answer"},
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "minion_answer",
				Description: openai.String("Minion answer with citation and explanation"),
				Schema:      schema,
				Strict:      openai.Bool(true),
			},
		},
	}
}

func jsonSchemaAggregateResponseFormat() openai.ChatCompletionNewParamsResponseFormatUnion {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"final":  map[string]any{"type": "boolean"},
			"answer": map[string]any{"type": "string"},
		},
		"required": []string{"final", "answer"},
	}
	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "aggregate_answer",
				Description: openai.String("Aggregate answer with final flag"),
				Schema:      schema,
				Strict:      openai.Bool(true),
			},
		},
	}
}
