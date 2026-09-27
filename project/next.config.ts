import path from "node:path";
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // The app lives in project/ of the code repository; trace files from here, not from a parent lockfile.
  outputFileTracingRoot: path.join(__dirname),
};

export default nextConfig;
