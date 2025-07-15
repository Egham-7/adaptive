package minions

import (
	"adaptive-backend/internal/models"
	"adaptive-backend/internal/services/providers/provider_interfaces"
	"adaptive-backend/internal/utils"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
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
	minionModel string,
) (*openai.ChatCompletion, error) {
	const maxRounds = 5

	userQuery := getUserQuery(req)
	if userQuery == "" {
		return nil, errors.New("no user query found")
	}

	var previousResults []string

	for range maxRounds {
		// Step 1: Decompose into instructions
		instructions, err := s.remoteDecompose(ctx, remoteProv, userQuery, previousResults, req.Model)
		if err != nil {
			return nil, fmt.Errorf("decomposition failed: %w", err)
		}

		if len(instructions) == 0 {
			return nil, errors.New("no instructions generated")
		}

		// Step 2: Execute instructions in parallel
		results := s.executeInstructionsParallel(ctx, localProv, instructions, minionModel)

		// Step 3: Aggregate results
		response, isComplete, err := s.remoteAggregate(ctx, remoteProv, userQuery, results, req.Model)
		if err != nil {
			return nil, fmt.Errorf("aggregation failed: %w", err)
		}

		if isComplete {
			return response, nil
		}

		// Prepare for next round
		previousResults = extractResultsForNextRound(results)
	}

	return nil, errors.New("MinionS protocol did not converge within maximum rounds")
}

// OrchestrateMinionSStream runs the MinionS protocol loop (streaming).
func (s *MinionsOrchestrationService) OrchestrateMinionSStream(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	localProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	minionModel string,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	const maxRounds = 5

	userQuery := getUserQuery(req)
	if userQuery == "" {
		return nil, errors.New("no user query found")
	}

	var previousResults []string

	for range maxRounds {
		// Step 1: Decompose into instructions
		instructions, err := s.remoteDecompose(ctx, remoteProv, userQuery, previousResults, req.Model)
		if err != nil {
			return nil, fmt.Errorf("decomposition failed: %w", err)
		}

		if len(instructions) == 0 {
			return nil, errors.New("no instructions generated")
		}

		// Step 2: Execute instructions in parallel
		results := s.executeInstructionsParallel(ctx, localProv, instructions, minionModel)

		// Step 3: Aggregate results (streaming)
		stream, isComplete, err := s.remoteAggregateStream(ctx, remoteProv, userQuery, results, req.Model)
		if err != nil {
			return nil, fmt.Errorf("streaming aggregation failed: %w", err)
		}

		if isComplete {
			return stream, nil
		}

		// Prepare for next round
		previousResults = extractResultsForNextRound(results)
	}

	return nil, errors.New("MinionS streaming protocol did not converge within maximum rounds")
}

// InstructionResult represents the result of executing an instruction
type InstructionResult struct {
	Index       int
	Instruction string
	Result      string
	Success     bool
	Error       error
	NeedsRedraft bool
}

// WorkerPool manages a pool of workers for executing instructions
type WorkerPool struct {
	workerCount int
	jobChan     chan WorkerJob
	resultChan  chan *InstructionResult
	wg          sync.WaitGroup
}

// WorkerJob represents a job to be executed by a worker
type WorkerJob struct {
	Index       int
	Instruction string
	Ctx         context.Context
	LocalProv   provider_interfaces.LLMProvider
	MinionModel string
}

// remoteDecompose asks the remote LLM to break down the query into atomic instructions
func (s *MinionsOrchestrationService) remoteDecompose(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	userQuery string,
	previousResults []string,
	model string,
) ([]string, error) {
	systemPrompt := `You are an expert at breaking down complex queries into simple, atomic instructions.
Analyze the user query and decompose it into specific, actionable instructions that can be executed independently.
Each instruction should be clear, focused, and require no additional context.
Return your response as JSON with an array of instruction strings.`

	var userPrompt strings.Builder
	userPrompt.WriteString("User Query: ")
	userPrompt.WriteString(userQuery)

	if len(previousResults) > 0 {
		userPrompt.WriteString("\n\nPrevious Results:\n")
		for i, result := range previousResults {
			userPrompt.WriteString(fmt.Sprintf("%d. %s\n", i+1, result))
		}
		userPrompt.WriteString("\nBased on these previous results, what additional instructions are needed to fully answer the query?")
	}

	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(userPrompt.String()),
		},
		Model:          model,
		ResponseFormat: s.createDecomposeSchema(),
		Temperature:    openai.Float(0.1),
	}

	resp, err := remoteProv.Chat().Completions().CreateCompletion(ctx, &param)
	if err != nil {
		return nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, errors.New("no response choices")
	}

	var parsed struct {
		Instructions []string `json:"instructions"`
	}

	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, fmt.Errorf("failed to parse instructions: %w", err)
	}

	return parsed.Instructions, nil
}

// NewWorkerPool creates a new worker pool with the specified number of workers
func NewWorkerPool(workerCount int) *WorkerPool {
	return &WorkerPool{
		workerCount: workerCount,
		jobChan:     make(chan WorkerJob, workerCount*2),
		resultChan:  make(chan *InstructionResult, workerCount*2),
	}
}

// Start initializes and starts the worker pool
func (wp *WorkerPool) Start(s *MinionsOrchestrationService) {
	for i := 0; i < wp.workerCount; i++ {
		wp.wg.Add(1)
		go wp.worker(s)
	}
}

// Stop shuts down the worker pool
func (wp *WorkerPool) Stop() {
	close(wp.jobChan)
	wp.wg.Wait()
	close(wp.resultChan)
}

// worker processes jobs from the job channel
func (wp *WorkerPool) worker(s *MinionsOrchestrationService) {
	defer wp.wg.Done()
	for job := range wp.jobChan {
		result := s.executeInstruction(job.Ctx, job.LocalProv, job.Instruction, job.MinionModel)
		result.Index = job.Index
		wp.resultChan <- result
	}
}

// executeInstructionsParallel executes multiple instructions concurrently using worker pool
func (s *MinionsOrchestrationService) executeInstructionsParallel(
	ctx context.Context,
	localProv provider_interfaces.LLMProvider,
	instructions []string,
	minionModel string,
) []*InstructionResult {
	workerFunc := func(ctx context.Context, instruction string) (string, error) {
		result := s.executeInstruction(ctx, localProv, instruction, minionModel)
		if !result.Success {
			return "", result.Error
		}
		return result.Result, nil
	}

	workerPool := utils.NewWorkerPool(int(math.Min(float64(len(instructions)), 10)), workerFunc)
	results := workerPool.ProcessJobs(ctx, instructions)

	// Convert to InstructionResult format
	instructionResults := make([]*InstructionResult, len(results))
	for i, result := range results {
		instructionResults[i] = &InstructionResult{
			Index:       result.Index,
			Instruction: instructions[result.Index],
			Result:      result.Data,
			Success:     result.Success,
			Error:       result.Error,
		}
	}

	return instructionResults
}

// executeSelectiveInstructions executes only specified instruction indices
func (s *MinionsOrchestrationService) executeSelectiveInstructions(
	ctx context.Context,
	localProv provider_interfaces.LLMProvider,
	instructions []string,
	indices []int,
	minionModel string,
	previousResults []*InstructionResult,
) []*InstructionResult {
	if len(indices) == 0 {
		return previousResults
	}

	workerFunc := func(ctx context.Context, instruction string) (string, error) {
		result := s.executeInstruction(ctx, localProv, instruction, minionModel)
		if !result.Success {
			return "", result.Error
		}
		return result.Result, nil
	}

	workerPool := utils.NewWorkerPool(int(math.Min(float64(len(indices)), 10)), workerFunc)
	
	// Convert previous results to utils.Result format
	prevResults := make([]utils.Result[string], len(previousResults))
	for i, result := range previousResults {
		prevResults[i] = utils.Result[string]{
			Index:   result.Index,
			Data:    result.Result,
			Success: result.Success,
			Error:   result.Error,
		}
	}

	results := workerPool.ProcessSelectiveJobs(ctx, instructions, indices, prevResults)

	// Convert back to InstructionResult format
	instructionResults := make([]*InstructionResult, len(results))
	for i, result := range results {
		instructionResults[i] = &InstructionResult{
			Index:       result.Index,
			Instruction: instructions[result.Index],
			Result:      result.Data,
			Success:     result.Success,
			Error:       result.Error,
		}
	}

	return instructionResults
}

// executeInstruction executes a single instruction
func (s *MinionsOrchestrationService) executeInstruction(
	ctx context.Context,
	localProv provider_interfaces.LLMProvider,
	instruction string,
	minionModel string,
) *InstructionResult {
	systemPrompt := `You are a helpful assistant. Execute the given instruction precisely and provide a clear, factual response.
If you cannot execute the instruction, explain why. Be concise but complete in your response.`

	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(systemPrompt),
			openai.UserMessage(instruction),
		},
		Temperature: openai.Float(0.3),
	}

	// Only set model if provided (for HuggingFace, model is embedded in BaseURL)
	if minionModel != "" {
		param.Model = shared.ChatModel(minionModel)
	}

	resp, err := localProv.Chat().Completions().CreateCompletion(ctx, &param)
	if err != nil {
		return &InstructionResult{
			Instruction: instruction,
			Success:     false,
			Error:       err,
		}
	}

	if len(resp.Choices) == 0 {
		return &InstructionResult{
			Instruction: instruction,
			Success:     false,
			Error:       errors.New("no response choices"),
		}
	}

	return &InstructionResult{
		Instruction: instruction,
		Result:      resp.Choices[0].Message.Content,
		Success:     true,
	}
}

// remoteAggregate combines results and determines if the response is complete
func (s *MinionsOrchestrationService) remoteAggregate(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	userQuery string,
	results []*InstructionResult,
	model string,
) (*openai.ChatCompletion, bool, error) {
	aggregationPrompt := s.buildAggregationPrompt(userQuery, results)

	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(`You are an expert aggregator. Combine the instruction results to answer the user's query.
Determine if you have enough information to provide a complete answer.
Return JSON with 'complete' (boolean) and 'answer' (string) fields.`),
			openai.UserMessage(aggregationPrompt),
		},
		Model:          model,
		ResponseFormat: s.createAggregateSchema(),
		Temperature:    openai.Float(0.2),
	}

	resp, err := remoteProv.Chat().Completions().CreateCompletion(ctx, &param)
	if err != nil {
		return nil, false, err
	}

	if len(resp.Choices) == 0 {
		return nil, false, errors.New("no response choices")
	}

	var parsed struct {
		Complete bool   `json:"complete"`
		Answer   string `json:"answer"`
	}

	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, false, fmt.Errorf("failed to parse aggregation: %w", err)
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

	return finalResp, parsed.Complete, nil
}

// remoteAggregateStream performs streaming aggregation
func (s *MinionsOrchestrationService) remoteAggregateStream(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	userQuery string,
	results []*InstructionResult,
	model string,
) (*ssestream.Stream[openai.ChatCompletionChunk], bool, error) {
	aggregationPrompt := s.buildAggregationPrompt(userQuery, results)

	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(`You are an expert aggregator. Combine the instruction results to answer the user's query.
Determine if you have enough information to provide a complete answer.
Return JSON with 'complete' (boolean) and 'answer' (string) fields.`),
			openai.UserMessage(aggregationPrompt),
		},
		Model:          model,
		ResponseFormat: s.createAggregateSchema(),
		Temperature:    openai.Float(0.2),
	}

	origStream, err := remoteProv.Chat().Completions().StreamCompletion(ctx, &param)
	if err != nil {
		return nil, false, err
	}

	var (
		chunks         []*openai.ChatCompletionChunk
		contentBuilder strings.Builder
	)

	for origStream.Next() {
		chunk := origStream.Current()
		chunks = append(chunks, &chunk)
		if len(chunk.Choices) > 0 {
			contentBuilder.WriteString(chunk.Choices[0].Delta.Content)
		}
	}

	if err := origStream.Err(); err != nil {
		return nil, false, err
	}

	var parsed struct {
		Complete bool   `json:"complete"`
		Answer   string `json:"answer"`
	}

	if err := json.Unmarshal([]byte(contentBuilder.String()), &parsed); err != nil {
		return nil, false, fmt.Errorf("failed to parse streaming aggregation: %w", err)
	}

	decoder := newMemoryChunkDecoder(chunks)
	stream := ssestream.NewStream[openai.ChatCompletionChunk](decoder, nil)

	return stream, parsed.Complete, nil
}

// Helper functions

func (s *MinionsOrchestrationService) buildAggregationPrompt(
	userQuery string,
	results []*InstructionResult,
) string {
	var sb strings.Builder

	sb.WriteString("Original User Query: ")
	sb.WriteString(userQuery)
	sb.WriteString("\n\nInstruction Results:\n")

	for i, result := range results {
		sb.WriteString(fmt.Sprintf("\nInstruction %d: %s\n", i+1, result.Instruction))
		if result.Success {
			sb.WriteString(fmt.Sprintf("Result: %s\n", result.Result))
		} else {
			sb.WriteString(fmt.Sprintf("Error: %v\n", result.Error))
		}
	}

	sb.WriteString("\nPlease aggregate these results to answer the original query. ")
	sb.WriteString("Determine if you have sufficient information for a complete answer.")

	return sb.String()
}

func extractResultsForNextRound(results []*InstructionResult) []string {
	var summaries []string
	for _, result := range results {
		if result.Success {
			summaries = append(summaries, result.Result)
		}
	}
	return summaries
}


func getUserQuery(req *models.ChatCompletionRequest) string {
	for _, msg := range req.Messages {
		if msg.OfUser != nil {
			return msg.OfUser.Content.OfString.Value
		}
	}
	return ""
}

func (s *MinionsOrchestrationService) createDecomposeSchema() openai.ChatCompletionNewParamsResponseFormatUnion {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"instructions": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "string",
				},
				"description": "Array of atomic instructions to execute",
			},
		},
		"required": []string{"instructions"},
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "instruction_decomposition",
				Description: openai.String("Decompose query into atomic instructions"),
				Schema:      schema,
				Strict:      openai.Bool(true),
			},
		},
	}
}

func (s *MinionsOrchestrationService) createAggregateSchema() openai.ChatCompletionNewParamsResponseFormatUnion {
	schema := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"complete": map[string]any{
				"type":        "boolean",
				"description": "Whether the answer is complete",
			},
			"answer": map[string]any{
				"type":        "string",
				"description": "The aggregated answer",
			},
		},
		"required": []string{"complete", "answer"},
	}

	return openai.ChatCompletionNewParamsResponseFormatUnion{
		OfJSONSchema: &shared.ResponseFormatJSONSchemaParam{
			JSONSchema: shared.ResponseFormatJSONSchemaJSONSchemaParam{
				Name:        "aggregated_response",
				Description: openai.String("Aggregated response with completion status"),
				Schema:      schema,
				Strict:      openai.Bool(true),
			},
		},
	}
}
