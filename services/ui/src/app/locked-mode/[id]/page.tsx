import { LockedMode } from "../components/locked-mode";

export default async function ShotPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/locked-mode/${id}`)
  const json_data = await data.json()
  return (
    <LockedMode data={json_data.saddle_coil_fft} />
  );
}