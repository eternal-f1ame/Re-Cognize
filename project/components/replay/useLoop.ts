"use client";

import { useCallback, useEffect, useRef, useState } from "react";

// One beat per query, each beat a fixed sequence of phases (ms spent in each).
export type Timeline = {
  beats: number;
  phases: readonly number[];
  intro: number;   // the empty stage before the first query
  hold: number;    // the finished stage before the loop restarts
  fade: number;    // the fade back to the empty stage
};

// beat -1 is the empty stage; within a beat, `phase` indexes Timeline.phases.
export type Clock = { beat: number; phase: number };

// Clock for an SVG loop. It runs only while the stage is on screen, the tab is visible and the
// reader has not paused it. Under reduced motion it starts paused on the finished stage.
// `epoch` changes on every restart so the stage remounts instead of animating back to the start.
export function useLoop(t: Timeline) {
  const ref = useRef<HTMLDivElement>(null);
  const [clock, setClock] = useState<Clock>({ beat: -1, phase: 0 });
  const [fading, setFading] = useState(false);
  const [epoch, setEpoch] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [inView, setInView] = useState(false);
  const [tabVisible, setTabVisible] = useState(true);

  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    if (mq.matches) {
      setPlaying(false);
      setClock({ beat: t.beats - 1, phase: t.phases.length - 1 });
    }
  }, [t.beats, t.phases.length]);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const io = new IntersectionObserver(([e]) => setInView(e.isIntersecting), { threshold: 0.2 });
    io.observe(el);
    return () => io.disconnect();
  }, []);

  useEffect(() => {
    const onChange = () => setTabVisible(document.visibilityState === "visible");
    document.addEventListener("visibilitychange", onChange);
    return () => document.removeEventListener("visibilitychange", onChange);
  }, []);

  useEffect(() => {
    if (!(playing && inView && tabVisible)) return;
    let delay: number;
    let next: () => void;
    const { beat, phase } = clock;
    const lastPhase = t.phases.length - 1;
    if (fading) {
      delay = t.fade;
      next = () => {
        setClock({ beat: -1, phase: 0 });
        setEpoch((e) => e + 1);
        setFading(false);
      };
    } else if (beat < 0) {
      delay = t.intro;
      next = () => setClock({ beat: 0, phase: 0 });
    } else if (phase < lastPhase) {
      delay = t.phases[phase];
      next = () => setClock({ beat, phase: phase + 1 });
    } else if (beat < t.beats - 1) {
      delay = t.phases[phase];
      next = () => setClock({ beat: beat + 1, phase: 0 });
    } else {
      delay = t.phases[phase] + t.hold;
      next = () => setFading(true);
    }
    const id = window.setTimeout(next, delay);
    return () => window.clearTimeout(id);
  }, [playing, inView, tabVisible, fading, clock, t]);

  const toggle = useCallback(() => setPlaying((p) => !p), []);
  const restart = useCallback(() => {
    setFading(false);
    setClock({ beat: -1, phase: 0 });
    setEpoch((e) => e + 1);
    setPlaying(true);
  }, []);

  // Phase of query k: -1 not reached, a phase index while it is the current query, Infinity once done.
  const at = useCallback(
    (k: number) => (k < clock.beat ? Infinity : k === clock.beat ? clock.phase : -1),
    [clock],
  );

  return { ref, clock, at, fading, epoch, playing, toggle, restart };
}
