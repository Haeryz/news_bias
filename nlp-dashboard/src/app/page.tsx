"use client";
import { useState } from "react";
import ClassificationForm from "@/components/ClassificationForm";
import ResultDisplay from "@/components/ResultDisplay";
import { ClassificationResult } from "@/types";

export default function Home() {
  const [result, setResult] = useState<ClassificationResult | null>(null);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen p-4">
      <div className="w-full max-w-lg space-y-6">
        <div className="text-center">
          <h1 className="text-4xl font-bold text-gray-800 dark:text-white">Republic vs Democrat Classification</h1>
        </div>
        <div className="flex flex-col items-center space-y-6">
          <ClassificationForm onResult={setResult} />
          <ResultDisplay result={result} />
        </div>
      </div>
    </div>
  );
}