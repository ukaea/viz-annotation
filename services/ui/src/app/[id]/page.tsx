import {ElmGraph} from "../components/elmGraph";

export default async function ShotPage({params} : { params: Promise<{id: string}> }) {
  const {id} = await params
  const data = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/backend-api?shot_id=${id}`)
  const json_data = await data.json()
  return (
    <ElmGraph elms={json_data.elms} data={json_data.dalpha} shot_id={id}/>
  );
}