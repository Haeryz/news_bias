// File 1: /news_bias/nlp-dashboard/src/app/api/classify/route.ts
import { NextResponse } from 'next/server';

export async function POST(request: Request) {
  try {
    const { text, includeLime = false, fastMode = true, numFeatures = 5 } = await request.json();
      // Choose endpoint based on LIME preference
    const endpoint = includeLime ? '/classify_with_explanation' : '/classify';
    const baseUrl = 'http://localhost:8000';
    
    // Prepare request body
    const requestBody: Record<string, unknown> = { text };
    if (includeLime) {
      requestBody.include_explanation = true;
      requestBody.fast_mode = fastMode;
      requestBody.num_features = numFeatures;
    }
    
    console.log(`Using endpoint: ${endpoint}, LIME: ${includeLime}`);
    
    const response = await fetch(`${baseUrl}${endpoint}`, {
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
    console.error('Proxy error:', error);
    return NextResponse.json({ 
      error: 'Failed to classify text',
      details: error instanceof Error ? error.message : 'Unknown error'
    }, { status: 500 });
  }
}