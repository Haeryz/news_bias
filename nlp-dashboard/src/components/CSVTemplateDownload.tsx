"use client";
import AnimatedContainer from './AnimatedContainer';

export default function CSVTemplateDownload() {
  const downloadTemplate = (type: 'simple' | 'with-labels') => {
    let csvContent = '';
    
    if (type === 'simple') {
      csvContent = `content
"Enter your news article text here"
"Add more articles, one per row"
"Make sure to include quotes around text with commas"`;
    } else {
      csvContent = `content,label
"The new policy proposal will benefit American families by reducing taxes.",2
"This partisan attack on our democracy represents a dangerous threat.",1
"The economic data shows steady growth in employment rates this quarter.",2
"Republican lawmakers are prioritizing corporate interests over families.",0
"Scientists have reached consensus on evidence-based policy making.",2`;
    }

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    const url = URL.createObjectURL(blob);
    link.setAttribute('href', url);
    link.setAttribute('download', type === 'simple' ? 'news_batch_template.csv' : 'news_batch_with_labels_template.csv');
    link.style.visibility = 'hidden';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <AnimatedContainer className="w-full">
      <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg border border-blue-200 dark:border-blue-800">
        <h3 className="text-lg font-semibold mb-3 text-blue-800 dark:text-blue-200">
          Download CSV Template
        </h3>
        <p className="text-sm text-blue-700 dark:text-blue-300 mb-4">
          Need help formatting your CSV file? Download a template to get started.
        </p>
        
        <div className="flex flex-col sm:flex-row gap-3">
          <button
            onClick={() => downloadTemplate('simple')}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
          >
            📄 Simple Template
            <div className="text-xs opacity-80 mt-1">Content only</div>
          </button>          <button
            onClick={() => downloadTemplate('with-labels')}
            className="flex-1 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
          >
            📊 Template with Labels
            <div className="text-xs opacity-80 mt-1">Content + expected labels</div>
          </button>
        </div>
          <div className="mt-3 text-xs text-blue-600 dark:text-blue-400 space-y-1">
          <p><strong>Label meanings:</strong> 0=Republican, 1=Liberal, 2=Neutral, 3=Other</p>
          <p><strong>Tip:</strong> Wrap text containing commas in quotes</p>
        </div>
      </div>
    </AnimatedContainer>
  );
}
