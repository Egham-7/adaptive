package utils

import (
	"context"
	"math"
	"sync"
)

// Job represents a generic job to be executed by a worker
type Job[T any] struct {
	Index   int
	Data    T
	Context context.Context
}

// Result represents the result of a job execution
type Result[T any] struct {
	Index   int
	Data    T
	Success bool
	Error   error
}

// WorkerFunc is a function type that processes a job and returns a result
type WorkerFunc[T, R any] func(ctx context.Context, data T) (R, error)

// WorkerPool manages a pool of workers for executing jobs
type WorkerPool[T, R any] struct {
	workerCount int
	jobChan     chan Job[T]
	resultChan  chan Result[R]
	wg          sync.WaitGroup
	workerFunc  WorkerFunc[T, R]
}

// NewWorkerPool creates a new worker pool with the specified number of workers
func NewWorkerPool[T, R any](workerCount int, workerFunc WorkerFunc[T, R]) *WorkerPool[T, R] {
	return &WorkerPool[T, R]{
		workerCount: workerCount,
		jobChan:     make(chan Job[T], workerCount*2),
		resultChan:  make(chan Result[R], workerCount*2),
		workerFunc:  workerFunc,
	}
}

// Start initializes and starts the worker pool
func (wp *WorkerPool[T, R]) Start() {
	for i := 0; i < wp.workerCount; i++ {
		wp.wg.Add(1)
		go wp.worker()
	}
}

// Stop shuts down the worker pool
func (wp *WorkerPool[T, R]) Stop() {
	close(wp.jobChan)
	wp.wg.Wait()
	close(wp.resultChan)
}

// SubmitJob submits a job to the worker pool
func (wp *WorkerPool[T, R]) SubmitJob(job Job[T]) {
	wp.jobChan <- job
}

// GetResult retrieves a result from the worker pool
func (wp *WorkerPool[T, R]) GetResult() Result[R] {
	return <-wp.resultChan
}

// ProcessJobs processes a slice of jobs and returns results in order
func (wp *WorkerPool[T, R]) ProcessJobs(ctx context.Context, jobs []T) []Result[R] {
	wp.Start()
	defer wp.Stop()

	// Submit jobs
	for i, job := range jobs {
		wp.SubmitJob(Job[T]{
			Index:   i,
			Data:    job,
			Context: ctx,
		})
	}

	// Collect results
	results := make([]Result[R], len(jobs))
	for i := 0; i < len(jobs); i++ {
		result := wp.GetResult()
		results[result.Index] = result
	}

	return results
}

// ProcessSelectiveJobs processes only specified job indices and merges with previous results
func (wp *WorkerPool[T, R]) ProcessSelectiveJobs(ctx context.Context, jobs []T, indices []int, previousResults []Result[R]) []Result[R] {
	if len(indices) == 0 {
		return previousResults
	}

	wp.Start()
	defer wp.Stop()

	// Submit jobs only for specified indices
	for _, idx := range indices {
		if idx < len(jobs) {
			wp.SubmitJob(Job[T]{
				Index:   idx,
				Data:    jobs[idx],
				Context: ctx,
			})
		}
	}

	// Collect results and merge with previous results
	results := make([]Result[R], len(previousResults))
	copy(results, previousResults)

	for i := 0; i < len(indices); i++ {
		result := wp.GetResult()
		if result.Index < len(results) {
			results[result.Index] = result
		}
	}

	return results
}

// worker processes jobs from the job channel
func (wp *WorkerPool[T, R]) worker() {
	defer wp.wg.Done()
	for job := range wp.jobChan {
		data, err := wp.workerFunc(job.Context, job.Data)
		result := Result[R]{
			Index:   job.Index,
			Data:    data,
			Success: err == nil,
			Error:   err,
		}
		wp.resultChan <- result
	}
}

