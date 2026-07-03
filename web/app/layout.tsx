import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "OfferCheck — Job Offer Due Diligence Agent",
  description:
    "Paste a job offer, JD, or company name. A skeptical research agent investigates and returns a verdict with red-flag evidence.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
