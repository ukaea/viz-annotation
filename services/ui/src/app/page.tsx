export default function Home() {
  return (
    <div className="w-screen h-screen flex items-center justify-center bg-gradient-to-br from-gray-200 via-gray-300 to-gray-400">
      <div className="p-6 bg-white/60 text-gray-800 rounded-lg shadow-lg backdrop-blur-sm">
        <h1 className="text-2xl font-bold mb-4">
          MAST feature tagging and validation UI
        </h1>
        <ul className="list-none list-inside space-y-2">
          <li>
            <a
              href="/elm"
              className="inline-block px-4 py-2 bg-white/60 text-gray-800 font-semibold rounded hover:bg-white/80"
            >
              ELM analysis
            </a>
          </li>
          <li>
            <a
              href="/disruption"
              className="inline-block px-4 py-2 bg-white/60 text-gray-800 font-semibold rounded hover:bg-white/80"
            >
              Disruption analysis
            </a>
          </li>
          <li>
            <a
              href="/locked-mode"
              className="inline-block px-4 py-2 bg-white/60 text-gray-800 font-semibold rounded hover:bg-white/80"
            >
              Locked mode analysis
            </a>
          </li>
        </ul>
      </div>
    </div>
  )
}
