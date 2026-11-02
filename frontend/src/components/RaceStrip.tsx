import type { TrackRecordRound } from "../types";

/**
 * Form over time, one cell per round — the kit WeekStrip's pattern adapted
 * to rounds, over the counted picks only (the same picks the headline
 * scores; the store computes the rows).
 *
 * Height is how many counted picks stood behind the round; colour is the
 * round's average probability error, on a fixed scale anchored to the 0.250
 * a model that always said 50% would average. A round with any pick recorded
 * after its session started is drawn HOLLOW — the outline carries the
 * colour, the middle stays empty — so the late rounds are visible in the
 * shape of the strip itself, not only in a footnote.
 */
const TONES = [
  { max: 0.125, bg: "bg-pr-win", border: "border-pr-win" },
  { max: 0.1875, bg: "bg-pr-lean", border: "border-pr-lean" },
  { max: Infinity, bg: "bg-pr-loss", border: "border-pr-loss" },
] as const;

const toneFor = (brier: number) => TONES.find((t) => brier <= t.max) ?? TONES[TONES.length - 1];

function cellLabel(r: TrackRecordRound): string {
  const name = r.race_name ?? `Round ${r.round}`;
  const timing =
    r.n_post_session === 0
      ? "every pick recorded before the session"
      : `${r.n_pre_session} recorded before the session, ${r.n_post_session} after it started`;
  return `${name}: average probability error ${r.brier.toFixed(3)} over ${r.n} picks — ${timing}`;
}

export function RaceStrip({ rounds }: { rounds: TrackRecordRound[] }) {
  const maxN = Math.max(1, ...rounds.map((r) => r.n));
  if (rounds.length === 0) {
    return <p className="text-sm text-pr-text-dim">No graded round yet.</p>;
  }
  return (
    <div>
      <ol className="flex h-24 items-end gap-1" aria-label="Probability error by round">
        {rounds.map((r) => {
          const tone = toneFor(r.brier);
          const late = r.n_post_session > 0;
          const h = 18 + (r.n / maxN) * 82;
          const label = cellLabel(r);
          return (
            <li
              key={`${r.season}-${r.round}`}
              data-testid="race-cell"
              data-late={late || undefined}
              title={label}
              className={`min-w-1.5 max-w-9 flex-1 rounded-t-pr ${
                late ? `border-2 ${tone.border} bg-transparent` : tone.bg
              }`}
              style={{ height: `${h}%` }}
            >
              <span className="sr-only">{label}</span>
            </li>
          );
        })}
      </ol>
      <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-pr-text-faint">
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-pr bg-pr-text-faint" />
          Every pick recorded before the session
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2 rounded-pr border-2 border-pr-text-faint bg-transparent" />
          Includes picks recorded after the session started
        </span>
        <span>Taller = more picks that round</span>
        <span>Colour = average probability error, greener is sharper; always saying 50% would average 0.250</span>
      </p>
    </div>
  );
}
