import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { RaceAccuracyEntry, TrackRecordPick, TrackRecordResponse } from "../types";
import { MARKET_LABELS, TIER_LABELS } from "../lib/glossary";
import { ErrorState, Skeleton, StatTile, pct, pctFine, record, ProbabilityBar, HowCounted } from "../predictor-ui";
import { RaceAccuracyTable } from "../components/RaceAccuracyTable";
import { RaceStrip } from "../components/RaceStrip";
import { EveryPicksTable } from "../components/EveryPicksTable";
import { CalibrationBlock } from "../predictor-ui";
import { InfoTooltip } from "../components/InfoTooltip";

const plural = (n: number, one: string, many: string) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;

/**
 * The per-market record, percent-forward.
 *
 * The big number is the right-calls share over EVERY recorded pick for the
 * market; beside it, smaller and labelled, the same market over only the
 * picks made before the session started. One component renders both figures
 * from the two ledgers the API sends, so the pair cannot drift into printing
 * the same headings with different arithmetic under them.
 */
function MarketLedger({
  rows,
  preRows,
}: {
  rows: TrackRecordResponse["by_market"];
  preRows: TrackRecordResponse["by_market"];
}) {
  const pre = new Map(preRows.map((r) => [r.market, r]));
  return (
    <div className="overflow-x-auto">
      <table data-testid="ledger" className="w-full min-w-[36rem] border-collapse text-sm">
        <thead>
          <tr className="border-b border-pr-rule text-left font-pr-display text-xs uppercase tracking-wide text-pr-text-dim">
            <th className="py-2 pr-3 font-semibold">Market</th>
            <th className="py-2 pr-3 text-right font-semibold">
              Right calls{" "}
              <InfoTooltip text="Right calls: the share of recorded picks that came true, out of every pick for this market." />
            </th>
            <th className="py-2 pr-3 text-right font-semibold">
              Before the session{" "}
              <InfoTooltip text="The same market over only the picks recorded before their session started — what the model said on the weekend, before anything ran." />
            </th>
            <th className="py-2 text-right font-semibold">
              Probability error{" "}
              <InfoTooltip text="Probability error (lower is better): the squared gap between the predicted chance and what actually happened; 0 is perfect." />
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const before = pre.get(row.market);
            return (
              <tr key={row.market} className="border-b border-pr-rule last:border-0">
                <td className="py-2 pr-3 text-pr-text">
                  <div>{MARKET_LABELS[row.market] ?? row.market}</div>
                  <div className="text-xs text-pr-text-faint">Stated {pctFine(row.avg_predicted_prob)} on average</div>
                </td>
                <td className="w-48 py-2 pr-3 text-right">
                  <span className="font-pr-display text-2xl font-semibold tabular-nums text-pr-text">
                    {pctFine(row.hit_rate)}
                  </span>
                  <div className="mt-1">
                    <ProbabilityBar
                      segments={[
                        { label: "Right", prob: row.hit_rate },
                        { label: "Wrong", prob: Math.max(0, 1 - row.hit_rate) },
                      ]}
                      minSegmentPx={6}
                    />
                  </div>
                  <div className="mt-1 text-xs tabular-nums text-pr-text-faint">
                    {row.n.toLocaleString("en-US")} picks
                  </div>
                </td>
                <td className="py-2 pr-3 text-right">
                  {before ? (
                    <>
                      <span className="text-base font-semibold tabular-nums text-pr-text">
                        {pctFine(before.hit_rate)}
                      </span>
                      <div className="mt-1 text-xs tabular-nums text-pr-text-faint">
                        {before.n.toLocaleString("en-US")} picks
                      </div>
                    </>
                  ) : (
                    <span className="text-pr-text-faint">—</span>
                  )}
                </td>
                <td className="py-2 text-right">
                  <span className="tabular-nums text-pr-text">{row.brier.toFixed(3)}</span>
                  {before && (
                    <div className="text-xs tabular-nums text-pr-text-faint">
                      {before.brier.toFixed(3)} before the session
                    </div>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

export function TrackRecordPage() {
  const [data, setData] = useState<TrackRecordResponse | null>(null);
  const [accuracy, setAccuracy] = useState<RaceAccuracyEntry[] | null>(null);
  const [error, setError] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  // The every-pick ledger is fetched only when its fold is first opened:
  // the rows are the one part of the payload that grows with the database,
  // and most readers never open it.
  const [picks, setPicks] = useState<TrackRecordPick[] | null>(null);
  const [picksError, setPicksError] = useState(false);
  const [picksRequested, setPicksRequested] = useState(false);

  useEffect(() => {
    setError(false);
    setData(null);
    setAccuracy(null);
    setPicks(null);
    setPicksError(false);
    setPicksRequested(false);
    // session_type="race" keeps race predictions apart from the qualifying
    // and sprint snapshots stored alongside them. The picks are withheld
    // from this first load (includePicks=false) and fetched on demand.
    Promise.all([api.trackRecord(undefined, "race", false), api.raceAccuracy("post_qualifying", "race")])
      .then(([record, byRace]) => {
        setData(record);
        setAccuracy(byRace);
      })
      .catch(() => setError(true));
  }, [reloadKey]);

  const fetchPicks = () => {
    setPicksError(false);
    api
      .trackRecordPicks(undefined, "race")
      .then(setPicks)
      .catch(() => setPicksError(true));
  };

  if (error) {
    return <ErrorState message="We couldn't load the track record. Check your connection and try again." onRetry={() => setReloadKey((k) => k + 1)} />;
  }
  if (!data || !accuracy) return <Skeleton label="Loading track record…" />;

  const grouped = data.by_market.reduce<Record<string, TrackRecordResponse["by_market"]>>((acc, row) => {
    (acc[row.tier] ??= []).push(row);
    return acc;
  }, {});
  // The headline counts every recorded pick, whenever it was made, so
  // `n_rebuilt_sessions` is no longer what was withheld -- it is how many
  // sessions had a pick recorded at or after their own start, and the
  // reconciliation against the pre-session subset beside it.
  const afterStart = data.n_rebuilt_sessions ?? 0;
  const preSession = data.n_pre_session ?? data.pre_session?.n_resolved ?? 0;
  const postSessionPicks = data.n_post_session_picks ?? Math.max(0, data.n_resolved - preSession);
  // The same ledger over the pre-session subset, grouped the same way.
  const preGrouped = (data.pre_session?.by_market ?? []).reduce<Record<string, TrackRecordResponse["by_market"]>>(
    (acc, row) => {
      (acc[row.tier] ??= []).push(row);
      return acc;
    },
    {},
  );
  const byRound = data.by_round ?? [];

  // The by-race table is a different view and keeps its own strict rule: it
  // judges a session only when its snapshot was written before it started.
  const counted = accuracy.filter((r) => !r.rebuilt);
  const sum = (f: (r: RaceAccuracyEntry) => number) => counted.reduce((n, r) => n + f(r), 0);
  const podiumOf = sum((r) => r.podium_of);
  const pointsOf = sum((r) => r.points_finish_of);

  return (
    <div className="flex flex-col gap-5">
      {/* Kevin's call, 2026-11-01: F1's record shows EVERY recorded
          prediction, including the ones recorded after their session
          started. There is no betting market to compare an F1 prediction
          against, so each pick is checked against what actually happened;
          the late ones are labelled wherever they appear, and the
          pre-session figure stands beside every headline number, so the
          two are never silently mixed. This section is that record. */}
      <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
        <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Track record</h2>
        <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
          Every recorded prediction for a race session, checked against what actually happened. There are
          no betting odds to compare a Formula 1 prediction against, so the comparison is the prediction
          against the result: the chance the model stated, and whether it happened. One counted pick per
          session, driver and market — the first one recorded; a re-run of the model on a session that has
          already run does not replace that pick or count a second time.
          {postSessionPicks > 0 &&
            ` Of these ${data.n_resolved.toLocaleString("en-US")} picks, ${postSessionPicks.toLocaleString("en-US")} were recorded at or after their session's start, and ${preSession.toLocaleString("en-US")} before it.`}
        </p>
        <p data-testid="timing-note" className="mb-4 max-w-3xl rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-2 text-sm text-pr-text-dim">
          {postSessionPicks === 0
            ? "Every pick in this ledger was recorded before its session started, so the before-the-session figure beside each market is the same number over the same picks."
            : "A pick recorded after the session started still counts, and is never presented as one made before it: every row the API sends carries both timestamps and the derived label. The figure beside each market is that same market over only the picks made before the session — what the model said on the weekend."}
        </p>
        {data.n_resolved === 0 && <p className="py-4 text-sm text-pr-text-dim">No resolved predictions yet. They appear as sessions are run.</p>}
        {preSession === 0 && data.n_resolved > 0 && (
          <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
            No pick has been recorded in time yet, so this figure is empty rather than zero. It fills in as
            sessions are snapshotted before they start.
          </p>
        )}

        {Object.entries(grouped).map(([tier, rows]) => (
          <div key={tier} className="mb-5 last:mb-0">
            <h3 className="mb-2 font-pr-display text-sm font-semibold uppercase tracking-wide text-pr-text-dim">{TIER_LABELS[tier] ?? tier}</h3>
            <MarketLedger rows={rows} preRows={preGrouped[tier] ?? []} />
          </div>
        ))}

        {data.calibration && Object.keys(data.calibration).length > 0 && (
          <div className="mt-6 border-t border-pr-rule pt-4">
            <CalibrationBlock
              series={Object.entries(data.calibration).map(([market, buckets]) => ({
                key: market,
                label: MARKET_LABELS[market] ?? market,
                buckets: (buckets ?? []).filter((b) => b.n > 0).map((b) => ({ bucket: b.bucket, n: b.n, correct: b.correct })),
              }))}
              markets={data.by_market.map((row) => ({
                key: row.market,
                label: MARKET_LABELS[row.market] ?? row.market,
                counted: { correct: Math.round(row.hit_rate * row.n), n: row.n },
              }))}
            />
          </div>
        )}
      </section>

      {byRound.length > 0 && (
        <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
          <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Race by race</h2>
          <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
            One cell per round, over the counted picks: how sharp the model&apos;s stated chances were that
            weekend, across every market. A hollow cell is a round whose picks were recorded after the
            session started.
          </p>
          <RaceStrip rounds={byRound} />
        </section>
      )}

      <section className="rounded-pr border border-pr-rule bg-pr-panel p-4 sm:p-5">
        <h2 className="font-pr-display text-2xl font-bold uppercase tracking-wide text-pr-text">Accuracy by race</h2>
        <p className="mb-4 max-w-2xl text-sm text-pr-text-dim">
          This table judges only sessions snapshotted before they ran.
          {afterStart > 0 &&
            ` ${plural(afterStart, "session has", "sessions have")} at least one pick recorded at or after the start — a re-run of the model, not a pre-session call.`}
          {" "}Each race compares the post-qualifying prediction with what happened: the predicted winner, and how many of the predicted top 3 and top 10 landed there. The full record, including the sessions this table skips, is the track record above.
        </p>

        {counted.length === 0 ? (
          accuracy.length > 0 && (
            <p className="mb-4 rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-2 text-sm text-pr-text">
              No races have a prediction snapshotted before they ran yet. The record starts with the next race.
            </p>
          )
        ) : counted.length < 3 ? (
          <p className="mb-4 rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-2 text-sm text-pr-text">
            Too early to judge: {counted.length} races scored
          </p>
        ) : (
          <div className="mb-4 grid grid-cols-3 gap-3">
            <div data-testid="winners-called">
              <StatTile label="Winners called" value={record(sum((r) => Math.min(r.win_hits, r.win_of)), sum((r) => r.win_of))} />
            </div>
            <div>
              <StatTile label="Podium places called" value={podiumOf ? pct(sum((r) => r.podium_hits) / podiumOf) : "—"} />
            </div>
            <div>
              <StatTile label="Points places called" value={pointsOf ? pct(sum((r) => r.points_finish_hits) / pointsOf) : "—"} />
            </div>
          </div>
        )}
        <RaceAccuracyTable entries={accuracy} />
      </section>

      <details
        className="rounded-pr border border-pr-rule bg-pr-panel px-4 py-3 text-sm"
        onToggle={(e) => {
          if ((e.currentTarget as HTMLDetailsElement).open && !picksRequested) {
            setPicksRequested(true);
            fetchPicks();
          }
        }}
      >
        <summary className="cursor-pointer font-semibold text-pr-text">Every recorded pick</summary>
        <div className="mt-3">
          {!picksRequested ? null : picksError ? (
            <div role="alert" className="flex flex-wrap items-center justify-between gap-3 rounded-pr border border-pr-loss/50 bg-pr-panel px-4 py-3 text-sm text-pr-text">
              <span>We couldn&apos;t load the recorded picks.</span>
              <button type="button" onClick={fetchPicks} className="rounded-pr border border-pr-rule bg-pr-panel-2 px-3 py-1.5 text-xs font-semibold text-pr-text hover:border-pr-accent">
                Try again
              </button>
            </div>
          ) : picks === null ? (
            <Skeleton label="Loading recorded picks…" />
          ) : (
            <>
              <p className="mb-3 max-w-2xl text-sm text-pr-text-dim">
                {picks.length.toLocaleString("en-US")} recorded picks, counted and not. The counted pick for a
                session, driver and market is the first one recorded; a later re-run is printed here marked
                Not counted and never replaces it.
              </p>
              <EveryPicksTable picks={picks} />
            </>
          )}
        </div>
      </details>

      <HowCounted moment="the session" headline="counted">
        <p>
          Predictions recorded after their session started are included in the figures above, labelled
          wherever they appear. That is the owner&apos;s decision (November 2026): this model&apos;s record is
          the whole record, and with no betting market to compare against, each prediction is judged against
          what actually happened, whenever it was recorded.
        </p>
        <p>
          A session can be recorded more than once when the model is re-run. The later rows are published in
          the ledger of every recorded pick, marked Not counted; they never replace the counted pick and are
          never scored.
        </p>
        <p>
          Every pick here is a yes-or-no chance, so nothing lands on a line: there are no pushes. A pick
          joins this record only once its session has been run and graded.
        </p>
      </HowCounted>
    </div>
  );
}
