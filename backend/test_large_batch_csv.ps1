# Test batch classification with larger dataset
# This script tests the batch classification endpoint using a larger test CSV file

# API URL for the batch classification endpoint
$url = "http://localhost:8000/upload_batch_csv"

# Path to the test CSV file
$filePath = Join-Path $PSScriptRoot "large_test_batch.csv"

# Ensure the file exists
if (-not (Test-Path $filePath)) {
    Write-Error "Test CSV file not found at: $filePath"
    exit 1
}

Write-Host "Testing batch classification with file: $filePath" -ForegroundColor Cyan

# Create form data with the file and parameters
$form = @{
    file = Get-Item -Path $filePath
    expected_labels = $true
    has_header = $true
}

try {
    # Send the request
    $response = Invoke-RestMethod -Uri $url -Method Post -Form $form -ContentType "multipart/form-data"
    
    # Display basic results
    Write-Host "`nBatch Processing Results:" -ForegroundColor Green
    Write-Host "Total processed: $($response.total_processed)"
    Write-Host "Successful predictions: $($response.successful_predictions)"
    Write-Host "Failed predictions: $($response.failed_predictions)"
    
    # Display evaluation metrics if available
    if ($response.evaluation) {
        Write-Host "`nEvaluation Metrics:" -ForegroundColor Green
        Write-Host "Accuracy: $($response.evaluation.accuracy.ToString("P2"))"
        Write-Host "Correct predictions: $($response.evaluation.correct_predictions) / $($response.evaluation.total_samples)"
        Write-Host "Incorrect predictions: $($response.evaluation.incorrect_predictions) / $($response.evaluation.total_samples)"
        
        # Get class report for each label
        Write-Host "`nClassification Report:" -ForegroundColor Green
        $report = $response.evaluation.classification_report
        
        foreach ($label in $response.evaluation.confusion_matrix.labels) {
            if ($report.$label) {
                $metrics = $report.$label
                Write-Host "$label - Precision: $($metrics.precision.ToString("F2")), Recall: $($metrics.recall.ToString("F2")), F1: $($metrics.f1-score.ToString("F2")), Support: $($metrics.support)"
            }
        }
        
        # Display overall metrics
        Write-Host "`nOverall Metrics:" -ForegroundColor Green
        Write-Host "Macro Avg - Precision: $($report.macro avg.precision.ToString("F2")), Recall: $($report.macro avg.recall.ToString("F2")), F1: $($report.macro avg.f1-score.ToString("F2"))"
        Write-Host "Weighted Avg - Precision: $($report.weighted avg.precision.ToString("F2")), Recall: $($report.weighted avg.recall.ToString("F2")), F1: $($report.weighted avg.f1-score.ToString("F2"))"
        
        # Display confusion matrix
        Write-Host "`nConfusion Matrix:" -ForegroundColor Green
        $matrix = $response.evaluation.confusion_matrix.matrix
        $labels = $response.evaluation.confusion_matrix.labels
        
        # Print header row
        Write-Host -NoNewline "Predicted →`nActual ↓  "
        foreach ($label in $labels) {
            Write-Host -NoNewline "$label`t"
        }
        Write-Host
        
        # Print matrix rows
        for ($i = 0; $i -lt $matrix.Count; $i++) {
            Write-Host -NoNewline "$($labels[$i])`t"
            foreach ($val in $matrix[$i]) {
                Write-Host -NoNewline "$val`t"
            }
            Write-Host
        }
        
        # Display sample of incorrect predictions (up to 5)
        $incorrectCount = [Math]::Min(5, $response.evaluation.incorrect_predictions)
        if ($incorrectCount -gt 0) {
            Write-Host "`nSample of Incorrect Predictions ($incorrectCount of $($response.evaluation.incorrect_predictions)):" -ForegroundColor Yellow
            $i = 0
            foreach ($item in $response.evaluation.incorrect_examples) {
                if ($i -ge $incorrectCount) { break }
                Write-Host "Text: $($item.content.Substring(0, [Math]::Min(100, $item.content.Length)))..."
                Write-Host "Expected: $($item.label) ($($labels[$item.label]))"
                Write-Host "Predicted: $($item.predicted_label) ($($labels[$item.predicted_label]))"
                Write-Host "Confidence: $($item."prob_$($labels[$item.predicted_label])")%"
                Write-Host "---"
                $i++
            }
        }
    }
    
    # Save detailed results to a JSON file
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = Join-Path $PSScriptRoot "large_batch_results_$timestamp.json"
    $response | ConvertTo-Json -Depth 10 | Out-File -FilePath $outputFile
    Write-Host "`nDetailed results saved to: $outputFile" -ForegroundColor Cyan
    
} catch {
    Write-Host "Error processing batch request:" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    
    if ($_.ErrorDetails) {
        $errorResponse = $_.ErrorDetails.Message | ConvertFrom-Json
        Write-Host "API Error: $($errorResponse.detail)" -ForegroundColor Red
    }
}
