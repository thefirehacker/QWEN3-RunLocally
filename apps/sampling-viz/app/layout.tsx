import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Qwen3 Sampling Visualizer",
  description: "Live visualization of temperature and top-p effects on token sampling",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
