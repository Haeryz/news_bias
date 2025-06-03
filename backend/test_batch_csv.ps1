# Test batch classification via CSV file upload
# This script tests the batch classification endpoint using a test CSV file

# API URL for the batch classification endpoint
$url = "http://localhost:8000/upload_batch_csv"

# Path to the test CSV file
$filePath = Join-Path $PSScriptRoot "test_batch.csv"

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
        
        # Display incorrect examples
        if ($response.evaluation.incorrect_predictions -gt 0) {
            Write-Host "`nIncorrect Predictions:" -ForegroundColor Yellow
            foreach ($item in $response.evaluation.incorrect_examples) {
                Write-Host "Text: $($item.content.Substring(0, [Math]::Min(100, $item.content.Length)))..."
                Write-Host "Expected: $($item.label) ($($labels[$item.label]))"
                Write-Host "Predicted: $($item.predicted_label) ($($labels[$item.predicted_label]))"
                Write-Host "Confidence: $($item."prob_$($labels[$item.predicted_label])")%"
                Write-Host "---"
            }
        }
    }
    
    # Save detailed results to a JSON file
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $outputFile = Join-Path $PSScriptRoot "batch_results_$timestamp.json"
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
