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

	"github.com/alitto/pond/v2"
	openai "github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/ssestream"
	"github.com/openai/openai-go/shared"
)

const maxRounds = 5

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
	userQuery, err := utils.ExtractLastMessage(req.Messages)
	if err != nil {
		return nil, errors.New("no user query found")
	}

	var (
		previousResults []string
		currentRound    *OrchestrationRound
	)

	for range maxRounds {
		// Step 1: Decompose into instructions (only if no current round or complete redraft needed)
		if currentRound == nil {
			instructions, err := s.remoteDecompose(ctx, remoteProv, userQuery, previousResults, req.Model)
			if err != nil {
				return nil, fmt.Errorf("decomposition failed: %w", err)
			}

			if len(instructions) == 0 {
				return nil, errors.New("no instructions generated")
			}

			// Step 2: Execute all instructions in parallel
			results := s.executeInstructionsParallel(ctx, localProv, instructions, minionModel)
			currentRound = &OrchestrationRound{
				Instructions: instructions,
				Results:      results,
			}
		}

		// Step 3: Aggregate results
		response, aggregation, err := s.remoteAggregate(ctx, remoteProv, userQuery, currentRound.Results, req.Model)
		if err != nil {
			return nil, fmt.Errorf("aggregation failed: %w", err)
		}

		if aggregation.Complete {
			return response, nil
		}

		// Step 4: Selective redrafting based on indices
		if len(aggregation.RedraftIndices) > 0 {
			// Re-execute only the specified indices
			for _, idx := range aggregation.RedraftIndices {
				if idx >= 0 && idx < len(currentRound.Instructions) {
					result := s.executeInstruction(ctx, localProv, currentRound.Instructions[idx], minionModel)
					result.Index = idx
					currentRound.Results[idx] = result
				}
			}
		} else {
			// If no specific indices to redraft, we can't improve further - exit
			break
		}
	}

	return nil, fmt.Errorf("MinionS protocol did not converge within %d rounds", maxRounds)
}

// OrchestrateMinionSStream runs the MinionS protocol loop (streaming).
func (s *MinionsOrchestrationService) OrchestrateMinionSStream(
	ctx context.Context,
	remoteProv provider_interfaces.LLMProvider,
	localProv provider_interfaces.LLMProvider,
	req *models.ChatCompletionRequest,
	minionModel string,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	// Run the entire MinionS protocol non-streaming to get final result
	finalResponse, err := s.OrchestrateMinionS(ctx, remoteProv, localProv, req, minionModel)
	if err != nil {
		return nil, err
	}

	// Extract the final answer from the response
	if len(finalResponse.Choices) == 0 {
		return nil, errors.New("no response choices in final result")
	}

	finalAnswer := finalResponse.Choices[0].Message.Content

	// Have a minion draft and stream the final response
	return s.streamFinalAnswer(ctx, localProv, finalAnswer, minionModel)
}

// InstructionResult represents the result of executing an instruction
type InstructionResult struct {
	Index       int
	Instruction string
	Result      string
	Success     bool
	Error       error
}

// OrchestrationRound represents a single round of the MinionS protocol
type OrchestrationRound struct {
	Instructions []string
	Results      []*InstructionResult
}

// AggregationResult contains the detailed response from aggregation
type AggregationResult struct {
	Complete       bool
	Answer         string
	RedraftIndices []int
	Feedback       string
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

// executeInstructionsParallel executes multiple instructions concurrently using pond worker pool
func (s *MinionsOrchestrationService) executeInstructionsParallel(
	ctx context.Context,
	localProv provider_interfaces.LLMProvider,
	instructions []string,
	minionModel string,
) []*InstructionResult {
	// Create a pond result pool with limited concurrency
	pool := pond.NewResultPool[*InstructionResult](int(math.Min(float64(len(instructions)), 10)), pond.WithContext(ctx))
	defer pool.StopAndWait()

	// Submit all instructions as tasks
	tasks := make([]pond.ResultTask[*InstructionResult], len(instructions))
	for i, instruction := range instructions {
		i, instruction := i, instruction // capture loop variables
		tasks[i] = pool.Submit(func() *InstructionResult {
			result := s.executeInstruction(ctx, localProv, instruction, minionModel)
			result.Index = i
			return result
		})
	}

	// Wait for all tasks to complete and collect results
	results := make([]*InstructionResult, len(instructions))
	for i, task := range tasks {
		result, err := task.Wait()
		if err != nil {
			// Handle task execution error
			results[i] = &InstructionResult{
				Index:       i,
				Instruction: instructions[i],
				Success:     false,
				Error:       err,
			}
		} else {
			results[i] = result
		}
	}

	return results
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
) (*openai.ChatCompletion, *AggregationResult, error) {
	aggregationPrompt := s.buildAggregationPrompt(userQuery, results)

	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage(`You are an expert aggregator. Combine the instruction results to answer the user's query.

IMPORTANT: Your role is to either:
1. APPROVE: If results are sufficient, set complete=true and provide the final answer
2. REQUEST REDRAFT: If results are insufficient, set complete=false and specify which instruction indices need redrafting

For incomplete answers, provide specific feedback and the exact indices that need improvement.`),
			openai.UserMessage(aggregationPrompt),
		},
		Model:          model,
		ResponseFormat: s.createAggregateSchema(),
		Temperature:    openai.Float(0.2),
	}

	resp, err := remoteProv.Chat().Completions().CreateCompletion(ctx, &param)
	if err != nil {
		return nil, nil, err
	}

	if len(resp.Choices) == 0 {
		return nil, nil, errors.New("no response choices")
	}

	var parsed struct {
		Complete       bool   `json:"complete"`
		Answer         string `json:"answer,omitempty"`
		RedraftIndices []int  `json:"redraft_indices,omitempty"`
		Feedback       string `json:"feedback,omitempty"`
	}

	if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &parsed); err != nil {
		return nil, nil, fmt.Errorf("failed to parse aggregation: %w", err)
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

	aggregation := &AggregationResult{
		Complete:       parsed.Complete,
		Answer:         parsed.Answer,
		RedraftIndices: parsed.RedraftIndices,
		Feedback:       parsed.Feedback,
	}

	return finalResp, aggregation, nil
}

// streamFinalAnswer creates a streaming response for the final answer using a minion
func (s *MinionsOrchestrationService) streamFinalAnswer(
	ctx context.Context,
	localProv provider_interfaces.LLMProvider,
	finalAnswer string,
	minionModel string,
) (*ssestream.Stream[openai.ChatCompletionChunk], error) {
	// Have a minion draft the final response in a streaming manner
	// This only happens AFTER the remote orchestrator has approved all chunks
	param := openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("You are a helpful assistant. The orchestrator has approved all instruction results. Draft a well-formatted final response based on the following approved aggregated information."),
			openai.UserMessage(finalAnswer),
		},
		Temperature: openai.Float(0.3),
	}

	// Only set model if provided (for HuggingFace, model is embedded in BaseURL)
	if minionModel != "" {
		param.Model = shared.ChatModel(minionModel)
	}

	return localProv.Chat().Completions().StreamCompletion(ctx, &param)
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
				"description": "Whether the answer is complete and ready for final approval",
			},
			"answer": map[string]any{
				"type":        "string",
				"description": "The aggregated answer (only if complete=true)",
			},
			"redraft_indices": map[string]any{
				"type": "array",
				"items": map[string]any{
					"type": "integer",
				},
				"description": "Array of instruction indices that need to be redrafted (only if complete=false)",
			},
			"feedback": map[string]any{
				"type":        "string",
				"description": "Feedback on what needs improvement for incomplete answers",
			},
		},
		"required": []string{"complete"},
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
