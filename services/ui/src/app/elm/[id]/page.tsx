import { ElmGraph } from "../components/elmGraph";

export default async function ShotPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/${id}`)
  const annotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}`)
  const json_data = await data.json()
  const json_annotations = await annotations.json()
  return (
    <ElmGraph elms={json_annotations.elms} data={json_data.dalpha} shot_id={id} />
  );
}