"use client";
import { useState, useEffect } from 'react';
import { ClassificationResult } from '@/types';
import AnimatedContainer from './AnimatedContainer';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

interface ResultDisplayProps {
  result: ClassificationResult | null;
}

export default function ResultDisplay({ result }: ResultDisplayProps) {
  const [animate, setAnimate] = useState(false);
  
  useEffect(() => {
    if (result) {
      setAnimate(false);
      setTimeout(() => setAnimate(true), 50);
    }
  }, [result]);

  if (!result) return null;

  // Ensure proper label mapping
  const getLabelMeaning = (label: number): string => {
    const labelMap: { [key: number]: string } = {
      0: 'Republican',
      1: 'Democrat', 
      2: 'Neutral',
      3: 'Others'
    };
    return labelMap[label] || 'Unknown';
  };

  // Use the mapped label meaning
  const displayLabel = getLabelMeaning(result.label);

  const getColorClass = () => {
    switch (displayLabel.toLowerCase()) {
      case 'republican':
        return 'border-red-500 bg-red-50 text-red-800 dark:bg-red-900/30 dark:text-red-200';
      case 'democrat':
        return 'border-blue-500 bg-blue-50 text-blue-800 dark:bg-blue-900/30 dark:text-blue-200';
      case 'neutral':
        return 'border-green-500 bg-green-50 text-green-800 dark:bg-green-900/30 dark:text-green-200';
      case 'others':
        return 'border-purple-500 bg-purple-50 text-purple-800 dark:bg-purple-900/30 dark:text-purple-200';
      default:
        return 'border-gray-500 bg-gray-50 text-gray-800 dark:bg-gray-900/30 dark:text-gray-200';
    }
  };  const getBadgeColor = () => {
    switch (displayLabel.toLowerCase()) {
      case 'republican':
        return 'bg-red-500 text-white border-red-600 shadow-red-200 dark:bg-red-600 dark:border-red-700';
      case 'democrat':
        return 'bg-blue-500 text-white border-blue-600 shadow-blue-200 dark:bg-blue-600 dark:border-blue-700';
      case 'neutral':
        return 'bg-green-500 text-white border-green-600 shadow-green-200 dark:bg-green-600 dark:border-green-700';
      case 'others':
        return 'bg-purple-500 text-white border-purple-600 shadow-purple-200 dark:bg-purple-600 dark:border-purple-700';
      default:
        return 'bg-gray-500 text-white border-gray-600 shadow-gray-200 dark:bg-gray-600 dark:border-gray-700';
    }
  };  // Prepare confidence scores chart data
  const confidenceChartData = {
    labels: ['REPUBLICAN', 'DEMOCRAT', 'NEUTRAL', 'OTHERS'],
    datasets: [
      {
        label: 'Confidence %',
        data: [
          result.confidence_scores.label_0,
          result.confidence_scores.label_1,
          result.confidence_scores.label_2,
          result.confidence_scores.label_3,
        ],
        backgroundColor: [
          'rgba(239, 68, 68, 0.8)',
          'rgba(59, 130, 246, 0.8)',
          'rgba(16, 185, 129, 0.8)',
          'rgba(147, 51, 234, 0.8)',
        ],
        borderColor: [
          'rgba(239, 68, 68, 1)',
          'rgba(59, 130, 246, 1)',
          'rgba(16, 185, 129, 1)',
          'rgba(147, 51, 234, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };
  // Prepare sentiment chart data
  const sentimentChartData = {
    labels: ['Positive', 'Neutral', 'Negative'],
    datasets: [
      {
        data: [
          result.sentiment?.scores?.positive ? result.sentiment.scores.positive * 100 : 0,
          result.sentiment?.scores?.neutral ? result.sentiment.scores.neutral * 100 : 0,
          result.sentiment?.scores?.negative ? result.sentiment.scores.negative * 100 : 0,
        ],
        backgroundColor: [
          'rgba(16, 185, 129, 0.8)',
          'rgba(156, 163, 175, 0.8)',
          'rgba(239, 68, 68, 0.8)',
        ],
        borderColor: [
          'rgba(16, 185, 129, 1)',
          'rgba(156, 163, 175, 1)',
          'rgba(239, 68, 68, 1)',
        ],
        borderWidth: 2,
      },
    ],
  };
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top' as const,
        labels: {
          color: 'rgb(55, 65, 81)',
          font: {
            size: 14,
            weight: 'bold' as const,
          },
        },
      },
      tooltip: {
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 13,
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        ticks: {
          color: 'rgb(55, 65, 81)',
          font: {
            size: 13,
            weight: 'bold' as const,
          },
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.3)',
        },
      },
      x: {
        ticks: {
          color: 'rgb(55, 65, 81)',
          font: {
            size: 13,
            weight: 'bold' as const,
          },
        },
        grid: {
          color: 'rgba(156, 163, 175, 0.3)',
        },
      },
    },
  };
  const doughnutOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'right' as const,
        labels: {
          color: 'rgb(55, 65, 81)',
          font: {
            size: 13,
            weight: 'bold' as const,
          },
          padding: 20,
        },
      },
      tooltip: {
        titleFont: {
          size: 14,
        },
        bodyFont: {
          size: 13,
        },
      },
    },
  };

  return (
    <AnimatedContainer className="w-full max-w-6xl mx-auto">
      <div 
        className={`transition-all duration-500 ease-in-out ${
          animate ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-6'
        }`}
      >        {/* Main Result Header */}
        <div className={`p-8 border-l-4 rounded-lg shadow-lg mb-6 ${getColorClass()}`}>
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">
            <div className="flex-1">
              <h2 className="text-2xl font-bold mb-4">Classification Result</h2>
              <div className="flex flex-col sm:flex-row sm:items-center gap-4">
                <div className="text-center sm:text-left">
                  <p className="text-sm opacity-75 mb-2">Political Bias</p>                  <span className={`inline-block px-6 py-3 rounded-full text-2xl font-bold border-2 shadow-md ${getBadgeColor()}`}>
                    {displayLabel.toUpperCase()}
                  </span>
                </div>
                <div className="text-center sm:text-left">
                  <p className="text-sm opacity-75 mb-1">Confidence</p>
                  <span className="text-3xl font-bold">
                    {result.confidence.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
            <div className="text-center md:text-right">
              <p className="text-sm opacity-75 mb-1">Bias Score</p>
              <p className="text-4xl font-bold">{result.bias_score.toFixed(1)}</p>
              <div className="w-full bg-gray-200 rounded-full h-2 mt-2">
                <div 
                  className="bg-gradient-to-r from-blue-500 to-red-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${Math.min(result.bias_score, 100)}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>        {/* Charts Section */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          {/* Confidence Scores Chart */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
              Classification Confidence
            </h3>
            <div className="h-64">
              <Bar data={confidenceChartData} options={chartOptions} />
            </div>
          </div>

          {/* Sentiment Analysis Chart */}
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
              Sentiment Analysis
            </h3>            <div className="h-64">
              {result.sentiment && result.sentiment.scores ? (
                <Doughnut data={sentimentChartData} options={doughnutOptions} />
              ) : (
                <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                  {result.sentiment_analysis_error ? (
                    <div className="text-center">
                      <p className="text-sm">Sentiment analysis unavailable</p>
                      <p className="text-xs mt-1">Azure service error</p>
                    </div>
                  ) : (
                    <p className="text-sm">No sentiment data available</p>
                  )}
                </div>
              )}
            </div>
            {result.sentiment && (
              <div className="mt-4 text-center">
                <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                  result.sentiment.label === 'positive' ? 'bg-green-100 text-green-800' :
                  result.sentiment.label === 'negative' ? 'bg-red-100 text-red-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  Overall: {result.sentiment.label}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* LIME Explanation Section */}
        {result.has_explanation && result.explanation && (
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200 flex items-center">
              <span className="mr-2">🔍</span>
              LIME Explanation
              <span className={`ml-2 px-2 py-1 text-xs rounded-full ${
                result.explanation.explanation_quality === 'high' ? 'bg-green-100 text-green-800' :
                result.explanation.explanation_quality === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-red-100 text-red-800'
              }`}>
                {result.explanation.explanation_quality} quality
              </span>
            </h3>
            
            <div className="space-y-4">
              {/* Simple Explanation */}
              <div className="bg-blue-50 dark:bg-blue-900/30 p-4 rounded-lg border border-blue-200 dark:border-blue-700">
                <h4 className="font-semibold text-blue-800 dark:text-blue-200 mb-2">AI Explanation</h4>
                <p className="text-blue-700 dark:text-blue-300 text-sm">
                  {result.explanation.simple_explanation}
                </p>
              </div>

              {/* Key Influences */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Supporting Words */}
                <div className="bg-green-50 dark:bg-green-900/30 p-4 rounded-lg border border-green-200 dark:border-green-700">
                  <h4 className="font-semibold text-green-800 dark:text-green-200 mb-3 flex items-center">
                    <span className="mr-2">✅</span>
                    Supporting Words
                  </h4>
                  <div className="space-y-2">
                    {result.explanation.key_influences.supporting.map(([word, weight], index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-green-700 dark:text-green-300 font-medium">{word}</span>
                        <div className="flex items-center">
                          <div className="w-16 h-2 bg-green-200 dark:bg-green-700 rounded mr-2">
                            <div 
                              className="h-full bg-green-500 rounded" 
                              style={{ width: `${Math.abs(weight) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-green-600 dark:text-green-400">
                            {(weight * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Opposing Words */}
                <div className="bg-red-50 dark:bg-red-900/30 p-4 rounded-lg border border-red-200 dark:border-red-700">
                  <h4 className="font-semibold text-red-800 dark:text-red-200 mb-3 flex items-center">
                    <span className="mr-2">❌</span>
                    Opposing Words
                  </h4>
                  <div className="space-y-2">
                    {result.explanation.key_influences.opposing.map(([word, weight], index) => (
                      <div key={index} className="flex justify-between items-center">
                        <span className="text-red-700 dark:text-red-300 font-medium">{word}</span>
                        <div className="flex items-center">
                          <div className="w-16 h-2 bg-red-200 dark:bg-red-700 rounded mr-2">
                            <div 
                              className="h-full bg-red-500 rounded" 
                              style={{ width: `${Math.abs(weight) * 100}%` }}
                            ></div>
                          </div>
                          <span className="text-xs text-red-600 dark:text-red-400">
                            {(Math.abs(weight) * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>              {/* Highlighted Text */}
              {result.explanation.highlighted_text && result.explanation.highlighted_text.length > 0 && (
                <div className="bg-gray-50 dark:bg-gray-700 p-4 rounded-lg border border-gray-200 dark:border-gray-600">
                  <h4 className="font-semibold text-gray-800 dark:text-gray-200 mb-3">Word Importance Visualization</h4>
                  <div className="text-sm leading-relaxed">
                    {result.explanation.highlighted_text.map((item, index) => {
                      const getHighlightColor = (level: string) => {
                        switch (level) {
                          case 'high_positive': return '#bbf7d0'; // bright green
                          case 'medium_positive': return '#d1fae5'; // medium green
                          case 'low_positive': return '#ecfdf5'; // light green
                          case 'high_negative': return '#fecaca'; // bright red
                          case 'medium_negative': return '#fed7d7'; // medium red
                          case 'low_negative': return '#fef2f2'; // light red
                          default: return '#f3f4f6'; // neutral gray
                        }
                      };                      const getOpacity = (level: string) => {
                        const baseOpacity = 0.3;
                        const intensityMap = {
                          'high_positive': 0.8,
                          'medium_positive': 0.6,
                          'low_positive': 0.4,
                          'high_negative': 0.8,
                          'medium_negative': 0.6,
                          'low_negative': 0.4,
                          'neutral': 0.1
                        };
                        return baseOpacity + (intensityMap[level as keyof typeof intensityMap] || 0.1);
                      };

                      return (
                        <span
                          key={index}
                          className="inline-block mr-1 mb-1 px-1 rounded"                          style={{
                            backgroundColor: getHighlightColor(item.highlight_level),
                            opacity: getOpacity(item.highlight_level)
                          }}
                          title={`Word: ${item.clean_word}, Importance: ${(item.importance * 100).toFixed(1)}%, Level: ${item.highlight_level}`}
                        >
                          {item.word}
                        </span>
                      );
                    })}
                  </div>
                  <div className="mt-3 text-xs text-gray-500 dark:text-gray-400">
                    💡 Hover over words to see their importance scores. Green = supports classification, Red = opposes classification
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Key Phrases Section */}
        {result.key_phrases && result.key_phrases.length > 0 && (
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg mb-6">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
              Top 5 Key Phrases
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-3">
              {result.key_phrases.slice(0, 5).map((phrase, index) => (
                <div
                  key={index}
                  className="bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-700 
                            rounded-lg p-3 text-center hover:bg-blue-100 dark:hover:bg-blue-900/50 
                            transition-colors duration-200"
                >
                  <span className="text-sm font-medium text-blue-800 dark:text-blue-200">
                    {phrase}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Highlighted Phrases Section */}
        {result.highlighted_phrases && result.highlighted_phrases.length > 0 && (
          <div className="bg-white dark:bg-gray-800 p-6 rounded-lg shadow-lg">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-gray-200">
              Highlighted Phrases Analysis
            </h3>
            <div className="space-y-4">
              {result.highlighted_phrases.map((phrase, index) => (
                <div
                  key={index}
                  className="border border-gray-200 dark:border-gray-600 rounded-lg p-4 
                            hover:shadow-md transition-shadow duration-200"
                >
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-2 mb-2">
                    <span className={`px-2 py-1 rounded text-xs font-semibold ${
                      phrase.sentiment === 'positive' ? 'bg-green-100 text-green-800' :
                      phrase.sentiment === 'negative' ? 'bg-red-100 text-red-800' :
                      'bg-gray-100 text-gray-800'
                    }`}>
                      {phrase.sentiment}
                    </span>                    <div className="text-xs text-gray-500 dark:text-gray-400">
                      Confidence: {(phrase.confidence_scores?.negative ? (phrase.confidence_scores.negative * 100).toFixed(1) : 0)}%
                    </div>
                  </div>                  <p className="text-sm text-gray-700 dark:text-gray-300 mb-2 italic">
                    &ldquo;{phrase.phrase}&rdquo;
                  </p>
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {phrase.explanation}
                  </p>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </AnimatedContainer>
  );
}