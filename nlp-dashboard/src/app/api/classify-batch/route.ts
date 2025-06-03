import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const expectedLabels = formData.get('expectedLabels') === 'true';
    const hasHeader = formData.get('hasHeader') === 'true';
    const saveReport = formData.get('saveReport') === 'true';
    const exportResults = formData.get('exportResults') === 'true';

    if (!file) {
      return NextResponse.json({ error: 'No file provided' }, { status: 400 });
    }

    if (!file.name.toLowerCase().endsWith('.csv')) {
      return NextResponse.json({ error: 'Only CSV files are accepted' }, { status: 400 });
    }

    // Read file content
    const fileContent = await file.text();

    const baseUrl = 'http://localhost:8000';
    
    // Prepare request body for backend
    const requestBody = {
      file_content: fileContent,
      expected_labels: expectedLabels,
      has_header: hasHeader,
      save_report: saveReport,
      export_results: exportResults
    };

    console.log('Sending batch request to backend:', { 
      filename: file.name, 
      size: fileContent.length,
      options: { expectedLabels, hasHeader, saveReport, exportResults }
    });

    const response = await fetch(`${baseUrl}/classify_batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody),
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error('Backend error:', response.status, errorText);
      throw new Error(`Backend responded with ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Batch processing error:', error);
    return NextResponse.json({ 
      error: 'Failed to process batch',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}
