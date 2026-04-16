import type { Metadata } from "next";
import { IBM_Plex_Mono } from "next/font/google";
import "./globals.css";

const ibmPlexMono = IBM_Plex_Mono({
  weight: ["400", "500", "700"],
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "KG-MAG — Knowledge-Grounded Article Generator",
  description: "Generate high-quality, cited Medium articles from your document corpus using RAG + LLM.",
  keywords: ["RAG", "article generator", "AI writing", "knowledge base"],
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={ibmPlexMono.variable}>
      <body className="antialiased">{children}</body>
    </html>
  );
}
