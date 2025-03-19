import {ElmGraph} from "../components/elmGraph";

export default async function ShotPage({params} : { params: Promise<{id: string}> }) {
  const {id} = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/${id}`)
  const annotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}?method=classic`)
  const modelAnnotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}?method=unet`);
  const json_data = await data.json()
  const json_annotations = await annotations.json()
  const json_model_annotations = await modelAnnotations.json()
  return (
    <ElmGraph model_elms={json_model_annotations.elms} elms={json_annotations.elms} data={json_data.dalpha} shot_id={id}/>
  );
}