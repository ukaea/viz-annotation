import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import ShotInput from "./components/shotInput";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "MAST Data Tagging",
  description: "A app for interactively tagging MAST data",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <div className="grid grid-cols-[200pt_1fr] gap-5 h-screen w-screen">
      <div className="bg-slate-200 p-5">
        <ShotInput />
      </div>
      <div>{children}</div>
    </div>
  );
}
