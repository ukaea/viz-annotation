'use client'
import { useEffect, useState } from "react";
import { ElmAnalysis } from "../components/elm-analysis";
import { useParams } from "next/navigation";
import { getShotData } from "../actions";

export default function ShotPage() {
  const params = useParams()
  const shot_id = params.id as string
  const [data, setData] = useState(null) // Look to make this type safe?
  
  /**
   * Makes use of server actions to allow client components to access the data
   * which facilitates clients requesting specific time ranges for example
   * based on a user input
   */
  useEffect(() => {
      const getData = async() => {
          const newData = await getShotData(shot_id)
          setData(newData)
      }
      getData()
  }, [shot_id])

  if (!data) return <div>Loading...</div>

  return (
    <ElmAnalysis elms={data.elms} data={data.dalpha} frequency={data.frequency} shot_id={shot_id}/>
  );
}