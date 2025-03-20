"use server";

export async function getShotData(shot_id: string) {
  const data = await fetch(
    `${process.env.NEXT_PUBLIC_API_URL}/backend-api?shot_id=${shot_id}`
  );
  return await data.json();
}
