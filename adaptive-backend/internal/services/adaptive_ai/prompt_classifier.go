package adaptive_ai

import (
	"adaptive-backend/internal/models"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/daulet/tokenizers"
	"github.com/yalue/onnxruntime_go"
)

var orderedOutputNames = []string{
	"task_type",
	"creativity_scope",
	"constraint_ct",
	"contextual_knowledge",
	"number_of_few_shots",
	"domain_knowledge",
	"no_label_reason",
	"reasoning",
}

// MAPPED: Task-specific weights for the final complexity score calculation.
var taskTypeWeights = map[string][]float32{
	"Open QA":         {0.2, 0.3, 0.15, 0.2, 0.15},
	"Closed QA":       {0.1, 0.35, 0.2, 0.25, 0.1},
	"Summarization":   {0.2, 0.25, 0.25, 0.1, 0.2},
	"Text Generation": {0.4, 0.2, 0.15, 0.1, 0.15},
	"Code Generation": {0.1, 0.3, 0.2, 0.3, 0.1},
	"Chatbot":         {0.25, 0.25, 0.15, 0.1, 0.25},
	"Classification":  {0.1, 0.35, 0.25, 0.2, 0.1},
	"Rewrite":         {0.2, 0.2, 0.3, 0.1, 0.2},
	"Brainstorming":   {0.5, 0.2, 0.1, 0.1, 0.1},
	"Extraction":      {0.05, 0.3, 0.3, 0.15, 0.2},
	"Other":           {0.25, 0.25, 0.2, 0.15, 0.15},
}

// PromptClassifierService provides local ONNX-based prompt classification
type PromptClassifierService struct {
	config models.PromptClassifierConfig
	// FIXED: The session type from the provided library is DynamicAdvancedSession.
	session     *onnxruntime_go.DynamicAdvancedSession
	modelConfig *models.ModelConfig
	tokenizer   *tokenizers.Tokenizer
	mu          sync.RWMutex
	initialized bool
	logger      *log.Logger

	// Store the retrieved input and output names for the session.
	inputNames      []string
	onnxOutputNames []string
	targetNames     []string
}

// NewPromptClassifierService creates a new local ONNX-based prompt classifier service
func NewPromptClassifierService(
	config models.PromptClassifierConfig,
	logger *log.Logger,
) (*PromptClassifierService, error) {
	if config.ModelID == "" {
		config.ModelID = "botirk/tiny-prompt-task-complexity-classifier"
	}
	if config.ModelPath == "" {
		config.ModelPath = "./models/prompt_classifier"
	}
	if config.Timeout == 0 {
		config.Timeout = 30 * time.Second
	}
	if config.MaxRetries == 0 {
		config.MaxRetries = 3
	}
	if config.MaxSeqLength == 0 {
		config.MaxSeqLength = 512
	}
	if logger == nil {
		logger = log.Default()
	}

	service := &PromptClassifierService{
		config: config,
		logger: logger,
	}

	return service, nil
}

// downloadModelFile and downloadModel are unchanged...
func (pcs *PromptClassifierService) downloadModelFile(
	filename,
	localPath string,
) error {
	url := fmt.Sprintf(
		"https://huggingface.co/%s/resolve/main/%s",
		pcs.config.ModelID,
		filename,
	)
	pcs.logger.Printf("Downloading %s from %s", filename, url)

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download %s: %w", filename, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf(
			"failed to download %s: HTTP %d",
			filename,
			resp.StatusCode,
		)
	}

	if err := os.MkdirAll(filepath.Dir(localPath), 0o755); err != nil {
		return fmt.Errorf("failed to create directory: %w", err)
	}

	out, err := os.Create(localPath)
	if err != nil {
		return fmt.Errorf("failed to create file %s: %w", localPath, err)
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file %s: %w", localPath, err)
	}

	pcs.logger.Printf("Successfully downloaded %s", filename)
	return nil
}

func (pcs *PromptClassifierService) downloadModel() error {
	files := []string{
		"model_quantized.onnx",
		"config.json",
		"tokenizer.json",
		"tokenizer_config.json",
		"special_tokens_map.json",
	}

	for _, file := range files {
		localPath := filepath.Join(pcs.config.ModelPath, file)
		if _, err := os.Stat(localPath); err == nil {
			pcs.logger.Printf("File %s already exists, skipping download", file)
			continue
		}

		if err := pcs.downloadModelFile(file, localPath); err != nil {
			if file == "model_quantized.onnx" || file == "config.json" ||
				file == "tokenizer.json" {
				return err // Critical files
			}
			pcs.logger.Printf(
				"Warning: failed to download optional file %s: %v",
				file,
				err,
			)
		}
	}
	return nil
}

func (pcs *PromptClassifierService) loadModelConfig() error {
	configPath := filepath.Join(pcs.config.ModelPath, "config.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read config.json: %w", err)
	}

	var config models.ModelConfig
	if err := json.Unmarshal(data, &config); err != nil {
		return fmt.Errorf("failed to parse config.json: %w", err)
	}

	var targetNames []string
	for name := range config.TargetSizes {
		targetNames = append(targetNames, name)
	}
	sort.Strings(targetNames)

	pcs.modelConfig = &config
	pcs.targetNames = targetNames
	pcs.logger.Printf(
		"Loaded model configuration with %d target heads",
		len(config.TargetSizes),
	)
	return nil
}

func (pcs *PromptClassifierService) Initialize(ctx context.Context) error {
	pcs.mu.Lock()
	defer pcs.mu.Unlock()

	if pcs.initialized {
		return nil
	}

	pcs.logger.Printf(
		"Initializing local prompt classifier with model: %s",
		pcs.config.ModelID,
	)

	if err := pcs.downloadModel(); err != nil {
		return fmt.Errorf("failed to download model: %w", err)
	}
	if err := pcs.loadModelConfig(); err != nil {
		return fmt.Errorf("failed to load model config: %w", err)
	}

	// FIXED: The library requires a global initialization.
	if !onnxruntime_go.IsInitialized() {
		err := onnxruntime_go.InitializeEnvironment()
		if err != nil {
			return fmt.Errorf("failed to initialize onnxruntime environment: %w", err)
		}
	}

	modelPath := filepath.Join(pcs.config.ModelPath, "model_quantized.onnx")

	// FIXED: Get input/output info before creating the session.
	inputs, outputs, err := onnxruntime_go.GetInputOutputInfo(modelPath)
	if err != nil {
		return fmt.Errorf("failed to get model input/output info: %w", err)
	}
	var inputNames, outputNames []string
	for _, i := range inputs {
		inputNames = append(inputNames, i.Name)
	}
	for _, o := range outputs {
		outputNames = append(outputNames, o.Name)
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(modelPath, inputNames, outputNames, nil)
	if err != nil {
		return fmt.Errorf("failed to create ONNX session: %w", err)
	}

	tokenizerPath := filepath.Join(pcs.config.ModelPath, "tokenizer.json")
	tk, err := tokenizers.FromFile(tokenizerPath)
	if err != nil {
		session.Destroy()
		return fmt.Errorf("failed to load tokenizer: %w", err)
	}

	pcs.session = session
	pcs.tokenizer = tk
	pcs.inputNames = inputNames
	pcs.onnxOutputNames = outputNames
	pcs.initialized = true
	pcs.logger.Println("Local prompt classifier initialized successfully")
	return nil
}

func (pcs *PromptClassifierService) IsInitialized() bool {
	pcs.mu.RLock()
	defer pcs.mu.RUnlock()
	return pcs.initialized
}

func (pcs *PromptClassifierService) ClassifyPrompt(
	ctx context.Context,
	prompt string,
) (*models.PromptClassificationResult, error) {
	if !pcs.IsInitialized() {
		if err := pcs.Initialize(ctx); err != nil {
			return nil, fmt.Errorf("failed to initialize service: %w", err)
		}
	}

	if prompt == "" {
		return nil, fmt.Errorf("prompt cannot be empty")
	}

	startTime := time.Now()
	result, err := pcs.classifyWithRetry(ctx, prompt)
	if err != nil {
		return nil, err
	}

	processingTime := time.Since(startTime).Milliseconds()
	result.ProcessingTimeMs = processingTime
	result.ModelVersion = pcs.config.ModelID

	pcs.logger.Printf(
		"Classified prompt locally in %dms: %s -> %s (%.3f)",
		processingTime,
		truncateString(prompt, 50),
		result.TaskType1,
		result.PromptComplexityScore,
	)
	return result, nil
}

// classifyWithRetry is unchanged...
func (pcs *PromptClassifierService) classifyWithRetry(
	ctx context.Context,
	prompt string,
) (*models.PromptClassificationResult, error) {
	var lastErr error
	for attempt := range pcs.config.MaxRetries + 1 {
		if attempt > 0 {
			delay := time.Duration(attempt) * time.Second
			pcs.logger.Printf(
				"Retrying classification attempt %d after %v",
				attempt+1,
				delay,
			)
			select {
			case <-time.After(delay):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}

		result, err := pcs.performLocalClassification(prompt)
		if err == nil {
			return result, nil
		}
		lastErr = err
		pcs.logger.Printf("Classification attempt %d failed: %v", attempt+1, err)
	}
	return nil, fmt.Errorf(
		"classification failed after %d attempts: %w",
		pcs.config.MaxRetries,
		lastErr,
	)
}

// performLocalClassification contains the main fixes for inference.
func (pcs *PromptClassifierService) performLocalClassification(
	prompt string,
) (*models.PromptClassificationResult, error) {
	// 1. Tokenize
	encoding := pcs.tokenizer.EncodeWithOptions(
		prompt,
		true,
		tokenizers.WithReturnAttentionMask(),
	)
	inputIDs := make([]int64, len(encoding.IDs))
	for i, v := range encoding.IDs {
		inputIDs[i] = int64(v)
	}
	attentionMask := make([]int64, len(encoding.AttentionMask))
	for i, v := range encoding.AttentionMask {
		attentionMask[i] = int64(v)
	}
	inputShape := onnxruntime_go.NewShape(1, int64(len(inputIDs)))

	inputTensor, err := onnxruntime_go.NewTensor(inputShape, inputIDs)
	if err != nil {
		return nil, fmt.Errorf("failed to create input_ids tensor: %w", err)
	}
	defer inputTensor.Destroy()
	maskTensor, err := onnxruntime_go.NewTensor(inputShape, attentionMask)
	if err != nil {
		return nil, fmt.Errorf("failed to create attention_mask tensor: %w", err)
	}
	defer maskTensor.Destroy()

	inputs := []onnxruntime_go.Value{inputTensor, maskTensor}
	outputs := make([]onnxruntime_go.Value, len(pcs.onnxOutputNames))

	err = pcs.session.Run(inputs, outputs)
	if err != nil {
		return nil, fmt.Errorf("onnx session run failed: %w", err)
	}
	defer func() {
		for _, o := range outputs {
			if o != nil {
				o.Destroy()
			}
		}
	}()

	batchResults, err := pcs.postProcessLogits(outputs)
	if err != nil {
		return nil, fmt.Errorf("failed to post-process logits: %w", err)
	}

	taskType := batchResults["task_type_1"].(string)
	weights, ok := taskTypeWeights[taskType]
	if !ok {
		weights = taskTypeWeights["Other"]
	}
	complexityScore := weights[0]*batchResults["creativity_scope"].(float32) +
		weights[1]*batchResults["reasoning"].(float32) +
		weights[2]*batchResults["constraint_ct"].(float32) +
		weights[3]*batchResults["domain_knowledge"].(float32) +
		weights[4]*batchResults["contextual_knowledge"].(float32)

	// 5. Assemble the final result struct
	finalResult := &models.PromptClassificationResult{
		TaskType1:             taskType,
		TaskType2:             batchResults["task_type_2"].(string),
		TaskTypeProbability:   batchResults["task_type_prob"].(float32),
		PromptComplexityScore: complexityScore,
		CreativityScope:       batchResults["creativity_scope"].(float32),
		Reasoning:             batchResults["reasoning"].(float32),
		ConstraintHandling:    batchResults["constraint_ct"].(float32),
		ContextualKnowledge:   batchResults["contextual_knowledge"].(float32),
		FewShotLearning:       batchResults["number_of_few_shots"].(float32),
		DomainKnowledge:       batchResults["domain_knowledge"].(float32),
		LabelReasoning:        batchResults["no_label_reason"].(float32),
	}

	return finalResult, nil
}

// postProcessLogits contains the fix for GetData().
func (pcs *PromptClassifierService) postProcessLogits(
	outputs []onnxruntime_go.Value,
) (map[string]any, error) {
	results := make(map[string]any)

	logitsMap := make(map[string][]float32)
	for i, name := range orderedOutputNames {
		if i >= len(outputs) {
			return nil, fmt.Errorf(
				"mismatch between ordered names (%d) and model outputs (%d)",
				len(orderedOutputNames),
				len(outputs),
			)
		}
		// FIXED: First, type-assert the Value interface to a concrete Tensor type.
		tensor, ok := outputs[i].(*onnxruntime_go.Tensor[float32])
		if !ok {
			// This can happen if the model output type is not float32.
			return nil, fmt.Errorf("output tensor %d (%s) is not a float32 tensor", i, name)
		}
		// FIXED: Then, call GetData() without a type assertion.
		logits := tensor.GetData()
		logitsMap[name] = logits
	}

	for _, targetName := range pcs.targetNames {
		logits, ok := logitsMap[targetName]
		if !ok {
			pcs.logger.Printf(
				"Warning: Logits for '%s' not found in model outputs. Skipping.",
				targetName,
			)
			continue
		}

		if targetName == "task_type" {
			task1, task2, prob := pcs.computeTaskTypeResults(logits)
			results["task_type_1"] = task1
			results["task_type_2"] = task2
			results["task_type_prob"] = prob
		} else {
			score := pcs.computeScoreResults(logits, targetName)
			results[targetName] = score
		}
	}
	return results, nil
}

// computeTaskTypeResults is unchanged...
func (pcs *PromptClassifierService) computeTaskTypeResults(
	preds []float32,
) (string, string, float32) {
	softmaxProbs := softmax(preds)
	top2Indices := argTopK(softmaxProbs, 2)

	primaryIdx := top2Indices[0]
	secondaryIdx := top2Indices[1]

	primaryProb := softmaxProbs[primaryIdx]
	secondaryProb := softmaxProbs[secondaryIdx]

	primaryString := pcs.modelConfig.TaskTypeMap[fmt.Sprint(primaryIdx)]
	secondaryString := pcs.modelConfig.TaskTypeMap[fmt.Sprint(secondaryIdx)]

	if secondaryProb < 0.1 {
		secondaryString = "NA"
	}

	return primaryString, secondaryString, primaryProb
}

// computeScoreResults is unchanged...
func (pcs *PromptClassifierService) computeScoreResults(
	preds []float32,
	target string,
) float32 {
	predsSoftmax := softmax(preds)
	weights := pcs.modelConfig.WeightsMap[target]
	divisor := pcs.modelConfig.DivisorMap[target]

	var weightedSum float32
	for i := range predsSoftmax {
		weightedSum += predsSoftmax[i] * weights[i]
	}

	score := weightedSum / divisor

	if target == "number_of_few_shots" && score < 0.05 {
		return 0.0
	}
	return score
}

// GetHealth and Shutdown are unchanged...
func (pcs *PromptClassifierService) GetHealth(
	ctx context.Context,
) *models.HealthStatus {
	health := &models.HealthStatus{
		Service:       "prompt_classifier_local",
		Timestamp:     time.Now().UTC(),
		ModelID:       pcs.config.ModelID,
		ModelPath:     pcs.config.ModelPath,
		Initialized:   pcs.IsInitialized(),
		InferenceType: "local_onnx",
	}

	if pcs.IsInitialized() {
		testCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
		defer cancel()

		start := time.Now()
		_, err := pcs.ClassifyPrompt(testCtx, "Health check test prompt")
		latency := time.Since(start)

		health.Status = "healthy"
		health.LatencyMs = latency.Milliseconds()

		if err != nil {
			health.Status = "degraded"
			health.Details = map[string]any{"error": err.Error()}
		}
	} else {
		health.Status = "not_initialized"
	}
	return health
}

func (pcs *PromptClassifierService) Shutdown() error {
	pcs.mu.Lock()
	defer pcs.mu.Unlock()

	pcs.logger.Println("Shutting down local prompt classifier service")
	if pcs.session != nil {
		pcs.session.Destroy()
		pcs.session = nil
	}
	if pcs.tokenizer != nil {
		pcs.tokenizer.Close()
		pcs.tokenizer = nil
	}
	// The provided library source implies a global environment.
	// Depending on application structure, you might call DestroyEnvironment() here
	// or when the entire application exits.
	// onnxruntime_go.DestroyEnvironment()
	pcs.initialized = false
	return nil
}

// --- Utility Functions ---

// argTopK is unchanged...
func argTopK(slice []float32, k int) []int {
	type pair struct {
		index int
		value float32
	}
	pairs := make([]pair, len(slice))
	for i, v := range slice {
		pairs[i] = pair{i, v}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].value > pairs[j].value
	})
	indices := make([]int, k)
	for i := range k {
		indices[i] = pairs[i].index
	}
	return indices
}

// softmax is unchanged...
func softmax(logits []float32) []float32 {
	if len(logits) == 0 {
		return []float32{}
	}
	probs := make([]float32, len(logits))
	var sum float64
	maxLogit := logits[0]
	for _, l := range logits {
		if l > maxLogit {
			maxLogit = l
		}
	}
	for i, l := range logits {
		p := math.Exp(float64(l - maxLogit))
		probs[i] = float32(p)
		sum += p
	}
	for i := range probs {
		probs[i] /= float32(sum)
	}
	return probs
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
