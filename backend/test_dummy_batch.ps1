# test_dummy_batch.ps1
# Test script for the batch classification endpoint using our dummy test dataset

Write-Host "Testing the dummy batch classification endpoint..." -ForegroundColor Cyan

$filePath = "dummy_test_batch.csv"
$fullPath = Join-Path -Path (Get-Location) -ChildPath $filePath

if (-not (Test-Path $fullPath)) {
    Write-Host "Error: File not found at $fullPath" -ForegroundColor Red
    exit 1
}

# Define the API endpoint URL - modify if needed
$apiUrl = "http://localhost:8000/upload_batch_csv"

# Read the file content
$fileBytes = [System.IO.File]::ReadAllBytes($fullPath)

# Create a boundary for the form data
$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

# Ask if the user wants to generate an HTML report and export results
$saveReport = Read-Host -Prompt "Generate HTML report? (y/n)"
$saveReport = ($saveReport -eq "y")

$exportResults = Read-Host -Prompt "Export results to CSV? (y/n)"  
$exportResults = ($exportResults -eq "y")

# Create the multipart form data content
$bodyLines = (
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"$filePath`"",
    "Content-Type: text/csv",
    "",
    [System.Text.Encoding]::UTF8.GetString($fileBytes),
    "--$boundary",
    "Content-Disposition: form-data; name=`"expected_labels`"",
    "",
    "true",
    "--$boundary",
    "Content-Disposition: form-data; name=`"has_header`"",
    "",
    "true",
    "--$boundary",
    "Content-Disposition: form-data; name=`"save_report`"",
    "",
    "$saveReport".ToLower(),
    "--$boundary",
    "Content-Disposition: form-data; name=`"export_results`"",
    "",
    "$exportResults".ToLower(),
    "--$boundary--"
)

$body = $bodyLines -join $LF

# Send the request
try {
    Write-Host "Sending request to $apiUrl..." -ForegroundColor Yellow
    $start = Get-Date
    
    $response = Invoke-RestMethod -Uri $apiUrl -Method Post -Body $body -ContentType "multipart/form-data; boundary=$boundary"
    
    $end = Get-Date
    $duration = ($end - $start).TotalSeconds
    
    Write-Host "Response received in $duration seconds" -ForegroundColor Green
    
    # Display summary statistics
    Write-Host "`nSummary Statistics:" -ForegroundColor Cyan
    Write-Host "Total processed: $($response.total_processed)" -ForegroundColor White
    Write-Host "Successful predictions: $($response.successful_predictions)" -ForegroundColor Green
    Write-Host "Failed predictions: $($response.failed_predictions)" -ForegroundColor Red
    
    # Display evaluation metrics if available
    if ($response.evaluation) {
        Write-Host "`nEvaluation Metrics:" -ForegroundColor Cyan
        Write-Host "Accuracy: $($response.evaluation.accuracy)" -ForegroundColor White
        
        Write-Host "`nClassification Report:" -ForegroundColor Cyan
        $report = $response.evaluation.classification_report
        
        # Display per-class metrics
        foreach ($className in $report.Keys) {
            if ($className -notmatch "accuracy|macro avg|weighted avg") {                $classMetrics = $report[$className]
                Write-Host "  $className - Precision: $([math]::Round($classMetrics.precision, 2)), Recall: $([math]::Round($classMetrics.recall, 2)), F1: $([math]::Round($classMetrics.'f1-score', 2))" -ForegroundColor White
            }
        }
        
        # Display confusion matrix
        Write-Host "`nConfusion Matrix:" -ForegroundColor Cyan
        $cm = $response.evaluation.confusion_matrix.matrix
        $labels = $response.evaluation.confusion_matrix.labels
        
        # Print header row with class labels
        Write-Host "True\Pred  " -NoNewline -ForegroundColor Gray
        foreach ($label in $labels) {
            Write-Host "$($label.PadRight(10))" -NoNewline -ForegroundColor Gray
        }
        Write-Host ""
        
        # Print each row of the confusion matrix
        for ($i = 0; $i -lt $cm.Count; $i++) {
            Write-Host "$($labels[$i].PadRight(10))" -NoNewline -ForegroundColor Gray
            foreach ($val in $cm[$i]) {
                if ($val -gt 0) {
                    Write-Host "$($val.ToString().PadRight(10))" -NoNewline -ForegroundColor Green
                } else {
                    Write-Host "$($val.ToString().PadRight(10))" -NoNewline -ForegroundColor Gray
                }
            }
            Write-Host ""
        }
        
        # Display incorrect predictions if any
        if ($response.evaluation.incorrect_predictions -gt 0) {
            Write-Host "`nIncorrect Predictions ($($response.evaluation.incorrect_predictions)):" -ForegroundColor Red
            foreach ($item in $response.evaluation.incorrect_examples) {
                $truncatedContent = $item.content
                if ($truncatedContent.Length -gt 100) {
                    $truncatedContent = $truncatedContent.Substring(0, 97) + "..."
                }
                Write-Host "  Text: $truncatedContent" -ForegroundColor White
                Write-Host "  Expected: $($item.label) ($($response.evaluation.confusion_matrix.labels[$item.label]))" -ForegroundColor Yellow
                Write-Host "  Predicted: $($item.predicted_label) ($($response.evaluation.confusion_matrix.labels[$item.predicted_label]))" -ForegroundColor Red
                Write-Host "  Bias score: $($item.bias_score)" -ForegroundColor Blue
                Write-Host ""
            }
        }
    }
    
    # Ask if the user wants to see the full JSON response
    $showFull = Read-Host -Prompt "Would you like to see the full JSON response? (y/n)"
    if ($showFull -eq "y") {
        $response | ConvertTo-Json -Depth 10
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody" -ForegroundColor Red
    }
}
