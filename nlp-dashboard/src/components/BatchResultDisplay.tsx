"use client";
import { useState } from 'react';
import { BatchClassificationResult } from '@/types';
import AnimatedContainer from './AnimatedContainer';

interface BatchResultDisplayProps {
  result: BatchClassificationResult;
  onClear: () => void;
}

export default function BatchResultDisplay({ result, onClear }: BatchResultDisplayProps) {
  const [showAllResults, setShowAllResults] = useState(false);
  const [selectedTab, setSelectedTab] = useState<'summary' | 'results' | 'evaluation'>('summary');

  const displayResults = showAllResults ? result.results : result.results.slice(0, 10);
  const hasEvaluation = result.evaluation !== undefined;

  // Calculate label distribution
  const labelDistribution = result.results.reduce((acc, item) => {
    const label = item.predicted_meaning || 'Unknown';
    acc[label] = (acc[label] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  return (
    <AnimatedContainer className="w-full">
      <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
        <div className="flex justify-between items-center mb-6">
          <h2 className="text-2xl font-bold text-gray-800 dark:text-gray-200">
            Batch Classification Results
          </h2>
          <button
            onClick={onClear}
            className="px-4 py-2 bg-gray-500 hover:bg-gray-600 text-white rounded-lg transition-colors"
          >
            Clear Results
          </button>
        </div>

        {/* Tab Navigation */}
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 p-1 rounded-lg mb-6">
          <button
            onClick={() => setSelectedTab('summary')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              selectedTab === 'summary'
                ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
            }`}
          >
            Summary
          </button>
          <button
            onClick={() => setSelectedTab('results')}
            className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
              selectedTab === 'results'
                ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
            }`}
          >
            Results ({result.results.length})
          </button>
          {hasEvaluation && (
            <button
              onClick={() => setSelectedTab('evaluation')}
              className={`flex-1 py-2 px-4 rounded-md font-medium transition-all ${
                selectedTab === 'evaluation'
                  ? 'bg-white dark:bg-gray-600 text-blue-600 dark:text-blue-400 shadow-sm'
                  : 'text-gray-600 dark:text-gray-300 hover:text-gray-800 dark:hover:text-gray-100'
              }`}
            >
              Evaluation
            </button>
          )}
        </div>

        {/* Summary Tab */}
        {selectedTab === 'summary' && (
          <div className="space-y-6">
            {/* Key Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {result.total_processed}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Total Articles</div>
              </div>
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {result.successful_predictions}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Successful</div>
              </div>
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-red-600 dark:text-red-400">
                  {result.failed_predictions}
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Failed</div>
              </div>
              {hasEvaluation && (
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                    {(result.evaluation!.accuracy * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-gray-600 dark:text-gray-300">Accuracy</div>
                </div>
              )}
            </div>

            {/* Processing Stats */}
            {result.processing_stats && (
              <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200">
                  Processing Performance
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <div className="font-medium text-gray-600 dark:text-gray-300">Average Time</div>
                    <div className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                      {result.processing_stats.avg_processing_time_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-600 dark:text-gray-300">Total Time</div>
                    <div className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                      {result.processing_stats.total_batch_time_seconds.toFixed(1)} s
                    </div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-600 dark:text-gray-300">Min Time</div>
                    <div className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                      {result.processing_stats.min_processing_time_ms.toFixed(1)} ms
                    </div>
                  </div>
                  <div>
                    <div className="font-medium text-gray-600 dark:text-gray-300">Max Time</div>
                    <div className="text-lg font-semibold text-gray-800 dark:text-gray-200">
                      {result.processing_stats.max_processing_time_ms.toFixed(1)} ms
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Label Distribution */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200">
                Prediction Distribution
              </h3>
              <div className="space-y-2">
                {Object.entries(labelDistribution).map(([label, count]) => (
                  <div key={label} className="flex justify-between items-center">
                    <span className="text-gray-700 dark:text-gray-300">{label}</span>
                    <div className="flex items-center space-x-2">
                      <div className="w-24 bg-gray-200 dark:bg-gray-600 rounded-full h-2">
                        <div
                          className="bg-blue-600 h-2 rounded-full"
                          style={{ width: `${(count / result.results.length) * 100}%` }}
                        ></div>
                      </div>
                      <span className="text-sm font-medium text-gray-800 dark:text-gray-200 w-12 text-right">
                        {count}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Export Information */}
            {(result.report_path || result.exported_csv) && (
              <div className="bg-yellow-50 dark:bg-yellow-900/20 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-2 text-yellow-800 dark:text-yellow-200">
                  Generated Files
                </h3>
                {result.report_path && (
                  <p className="text-sm text-yellow-700 dark:text-yellow-300 mb-1">
                    📄 HTML Report: {result.report_path}
                  </p>
                )}
                {result.exported_csv && (
                  <p className="text-sm text-yellow-700 dark:text-yellow-300">
                    📊 CSV Export: {result.exported_csv}
                  </p>
                )}
              </div>
            )}
          </div>
        )}

        {/* Results Tab */}
        {selectedTab === 'results' && (
          <div className="space-y-4">
            {/* Controls */}
            <div className="flex justify-between items-center">
              <div className="text-sm text-gray-600 dark:text-gray-300">
                Showing {displayResults.length} of {result.results.length} results
              </div>
              {result.results.length > 10 && (
                <button
                  onClick={() => setShowAllResults(!showAllResults)}
                  className="px-3 py-1 text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded hover:bg-blue-200 dark:hover:bg-blue-900/50"
                >
                  {showAllResults ? 'Show Less' : 'Show All'}
                </button>
              )}
            </div>

            {/* Results Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-600">
                    <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">#</th>
                    <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Content</th>
                    {hasEvaluation && (
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Expected</th>
                    )}
                    <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Predicted</th>
                    <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Bias Score</th>
                    {hasEvaluation && (
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Status</th>
                    )}
                  </tr>
                </thead>
                <tbody>
                  {displayResults.map((item, index) => (
                    <tr key={index} className="border-b border-gray-100 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-700">
                      <td className="p-2 text-gray-600 dark:text-gray-400">{index + 1}</td>
                      <td className="p-2 max-w-xs">
                        <div className="truncate text-gray-800 dark:text-gray-200" title={item.content}>
                          {item.content.length > 100 ? `${item.content.substring(0, 100)}...` : item.content}
                        </div>
                      </td>
                      {hasEvaluation && (
                        <td className="p-2 text-gray-700 dark:text-gray-300">
                          {item.expected_meaning || item.label}
                        </td>
                      )}
                      <td className="p-2">
                        <span className="px-2 py-1 rounded-full text-xs bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-200">
                          {item.predicted_meaning}
                        </span>
                      </td>
                      <td className="p-2 text-gray-700 dark:text-gray-300">
                        {item.bias_score?.toFixed(1)}
                      </td>
                      {hasEvaluation && (
                        <td className="p-2">
                          {item.prediction_status === 'correct' && (
                            <span className="text-green-600 dark:text-green-400">✓</span>
                          )}
                          {item.prediction_status === 'incorrect' && (
                            <span className="text-red-600 dark:text-red-400">✗</span>
                          )}
                          {item.prediction_status === 'missing' && (
                            <span className="text-gray-400">-</span>
                          )}
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Evaluation Tab */}
        {selectedTab === 'evaluation' && hasEvaluation && (
          <div className="space-y-6">
            {/* Overall Metrics */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {(result.evaluation!.accuracy * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Accuracy</div>
              </div>
              <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {(result.evaluation!.avg_precision * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Avg Precision</div>
              </div>
              <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {(result.evaluation!.avg_f1 * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-gray-600 dark:text-gray-300">Avg F1 Score</div>
              </div>
            </div>

            {/* Per-Class Metrics */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200">
                Per-Class Performance
              </h3>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-200 dark:border-gray-600">
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Class</th>
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Precision</th>
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Recall</th>
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">F1 Score</th>
                      <th className="text-left p-2 font-medium text-gray-700 dark:text-gray-300">Support</th>
                    </tr>
                  </thead>
                  <tbody>
                    {result.evaluation!.class_metrics.map((metric) => (
                      <tr key={metric.label} className="border-b border-gray-100 dark:border-gray-700">
                        <td className="p-2 font-medium text-gray-800 dark:text-gray-200">
                          {metric.label_name}
                        </td>
                        <td className="p-2 text-gray-700 dark:text-gray-300">
                          {(metric.precision * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-gray-700 dark:text-gray-300">
                          {(metric.recall * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-gray-700 dark:text-gray-300">
                          {(metric.f1 * 100).toFixed(1)}%
                        </td>
                        <td className="p-2 text-gray-700 dark:text-gray-300">
                          {metric.support}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Confusion Matrix */}
            <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg">
              <h3 className="text-lg font-semibold mb-3 text-gray-800 dark:text-gray-200">
                Confusion Matrix
              </h3>
              <div className="overflow-x-auto">
                <table className="text-sm">
                  <thead>
                    <tr>
                      <th className="p-2"></th>
                      <th className="p-2 text-center font-medium text-gray-700 dark:text-gray-300" colSpan={result.evaluation!.confusion_matrix.labels.length}>
                        Predicted
                      </th>
                    </tr>
                    <tr>
                      <th className="p-2"></th>
                      {result.evaluation!.confusion_matrix.labels.map((label) => (
                        <th key={label} className="p-2 text-center text-xs text-gray-600 dark:text-gray-400">
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.evaluation!.confusion_matrix.matrix.map((row, i) => (
                      <tr key={i}>
                        {i === 0 && (
                          <th className="p-2 text-center font-medium text-gray-700 dark:text-gray-300" rowSpan={result.evaluation!.confusion_matrix.matrix.length}>
                            True
                          </th>
                        )}
                        <th className="p-2 text-right text-xs text-gray-600 dark:text-gray-400">
                          {result.evaluation!.confusion_matrix.labels[i]}
                        </th>
                        {Array.isArray(row) ? row.map((cell, j) => (
                          <td key={j} className={`p-2 text-center ${i === j ? 'bg-green-100 dark:bg-green-900/30 font-semibold' : 'bg-white dark:bg-gray-600'}`}>
                            {cell}
                          </td>
                        )) : (
                          <td className="p-2 text-center">
                            {row}
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Incorrect Examples (if any) */}
            {result.evaluation!.incorrect_examples.length > 0 && (
              <div className="bg-red-50 dark:bg-red-900/20 p-4 rounded-lg">
                <h3 className="text-lg font-semibold mb-3 text-red-800 dark:text-red-200">
                  Sample Incorrect Predictions ({result.evaluation!.incorrect_examples.length} total)
                </h3>
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {result.evaluation!.incorrect_examples.slice(0, 5).map((item, index) => (
                    <div key={index} className="text-sm p-2 bg-white dark:bg-gray-700 rounded border border-red-200 dark:border-red-800">
                      <div className="text-gray-800 dark:text-gray-200 mb-1">
                        &ldquo;{item.content.length > 150 ? `${item.content.substring(0, 150)}...` : item.content}&rdquo;
                      </div>
                      <div className="flex space-x-4 text-xs">
                        <span className="text-green-600 dark:text-green-400">
                          Expected: {item.expected_meaning}
                        </span>
                        <span className="text-red-600 dark:text-red-400">
                          Predicted: {item.predicted_meaning}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </AnimatedContainer>
  );
}
