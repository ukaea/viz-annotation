import {ElmGraph} from "../components/elmGraph";

export default async function ShotPage({params} : { params: Promise<{id: string}> }) {
  const {id} = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/data/${id}`)
  const annotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}?method=classic`)
  const modelAnnotations = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api/annotations/${id}?method=unet&force=true`);
  const metaData = await fetch('https://mastapp.site/json/shots/30275')
  const json_data = await data.json()
  const json_annotations = await annotations.json()
  const json_model_annotations = await modelAnnotations.json()
  const json_metadata = await metaData.json()

  const metadata = {
    'timestamp': json_metadata.timestamp,
    'pre_description': json_metadata.preshot_description,
    'post_description': json_metadata.postshot_description
  }
  return (
    <ElmGraph metadata={metadata} model_elms={json_model_annotations.elms} elm_type={json_annotations.elm_type} elms={json_annotations.elms} data={json_data.dalpha} shot_id={id}/>
  );
}