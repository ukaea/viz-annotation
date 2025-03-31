'use client'

import { useRouter } from "next/navigation"
import { FormEvent, useState } from "react"

type ShotInputProps = {
  endpoint: string;
}

export default function ShotInput({ endpoint }: ShotInputProps) {
  const router = useRouter();
  const [shotId, setShotId] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (shotId) {
      router.push(`/${endpoint}/${shotId}`)
    }
  };

  return (
    <div>
      <h1>
        Enter a shot ID
      </h1>
      <form onSubmit={handleSubmit} className="flex space-x-1">
        <input
          type="number"
          value={shotId}
          onChange={(e) => setShotId(e.target.value)}
          placeholder="Shot ID..."
          required
          className="p-1 border rounded"
        />
        <button type="submit" className="p-1 bg-blue-500 text-white rounded">Go</button>
      </form>
    </div>
  )
}