import { Nav } from "../components/Nav";
import { Hero } from "../components/Hero";
import { Abstract } from "../components/Abstract";
import { Highlights } from "../components/Highlights";
import { Protocols } from "../components/Protocols";
import { CommitCondition } from "../components/CommitCondition";
import { ReCast } from "../components/ReCast";
import { Results } from "../components/Results";
import { Resources } from "../components/Resources";
import { Citation } from "../components/Citation";
import { Footer } from "../components/Footer";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Abstract />
        <Highlights />
        <Protocols />
        <CommitCondition />
        <ReCast />
        <Results />
        <Resources />
        <Citation />
      </main>
      <Footer />
    </>
  );
}
