import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { driverName } from "../lib/teamColors";
import { pctFine } from "../predictor-ui";
import type { TrackRecordPick } from "../types";

/**
 * Every recorded pick, in a table: the chance the model stated, what
 * happened, and when the pick was made.
 *
 * A pick recorded after its session started carries "Made after the
 * session" in its own column — labelled, never hidden, and never blended
 * into the before-session rows. A rerun that lost the earliest-pick contest
 * is printed too, marked "Not counted": it is part of the recorded history,
 * just not part of the scored record.
 */
export function EveryPicksTable({ picks }: { picks: TrackRecordPick[] }) {
  const rows = [...picks].sort(
    (a, b) =>
      b.season - a.season ||
      b.round - a.round ||
      a.tier.localeCompare(b.tier) ||
      a.driver_id.localeCompare(b.driver_id) ||
      a.market.localeCompare(b.market),
  );
  if (rows.length === 0) {
    return <p className="text-sm text-pr-text-dim">No recorded picks yet.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table data-testid="every-pick-table" className="w-full min-w-[44rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            <th className="py-2 pr-3 font-semibold">Race</th>
            <th className="py-2 pr-3 font-semibold">Driver</th>
            <th className="py-2 pr-3 font-semibold">Market</th>
            <th className="py-2 pr-3 text-right font-semibold">Stated chance</th>
            <th className="py-2 pr-3 font-semibold">Outcome</th>
            <th className="py-2 pr-3 font-semibold">Made</th>
            <th className="py-2 font-semibold">Counted</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((p) => (
            <tr
              key={`${p.season}-${p.round}-${p.session_type}-${p.tier}-${p.driver_id}-${p.market}-${p.snapshotted_at}`}
              data-testid="pick-row"
              className="border-b border-pr-rule last:border-0"
            >
              <td className="py-2 pr-3 text-pr-text">
                {p.race_name ?? `Round ${p.round}`}
                <div className="text-xs text-pr-text-faint">
                  Round {p.round} · {p.season}
                </div>
              </td>
              <td className="py-2 pr-3 text-pr-text">{driverName(p.driver_id)}</td>
              <td className="py-2 pr-3 text-pr-text">
                {MARKET_LABELS[p.market] ?? p.market}
                <div className="text-xs text-pr-text-faint">{TIER_LABELS[p.tier] ?? p.tier}</div>
              </td>
              <td className="py-2 pr-3 text-right tabular-nums text-pr-text">{pctFine(p.predicted_prob)}</td>
              <td className="py-2 pr-3 text-pr-text">{p.actual_outcome === 1 ? "Happened" : "Did not happen"}</td>
              <td className="py-2 pr-3">
                {p.made_before_session ? (
                  <span className="text-pr-text-dim">Before the session</span>
                ) : (
                  <span className="font-medium text-pr-text">Made after the session</span>
                )}
              </td>
              <td className="py-2 text-pr-text-dim">{p.counted ? "Counted" : "Not counted"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
