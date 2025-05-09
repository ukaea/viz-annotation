import { ElmGraph } from "../components/elmGraph";

export default async function ShotPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/${id}`)
  const annotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}?method=classic`)
  const json_data = await data.json()
  const json_annotations = await annotations.json()
  return (
    <ElmGraph elm_type={json_annotations.elm_type} elms={json_annotations.elms} data={json_data.dalpha} shot_id={id}/>
  );
}