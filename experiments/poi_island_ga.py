"""Frozen-pool PoI island genetic experiment. Never uses target draw for optimization.

Requires numpy. Input CSV columns:
draw_key,rank,main_numbers,auxiliary_numbers,score,created_utc
The score is higher-is-better, e.g. negative distance from the causal PoI G target.
Draw CSV: draw_key,ball_1..ball_5,star_1..star_2
Prize JSON: {"YYYY-MM-DD": {"2+1": 10.0, "5+2": 1000000, ...}}
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
import numpy as np

def ticket_of(row):
    m = tuple(sorted(json.loads(row["main_numbers"])))
    s = tuple(sorted(json.loads(row["auxiliary_numbers"])))
    if len(m) != 5 or len(set(m)) != 5 or not all(1 <= i <= 50 for i in m):
        raise ValueError("Invalid main combination")
    if len(s) != 2 or len(set(s)) != 2 or not all(1 <= i <= 12 for i in s):
        raise ValueError("Invalid star combination")
    return m, s

def read_pools(path, required_size):
    pools = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            draw = row["draw_key"]
            stamp = datetime.fromisoformat(row["created_utc"].replace("Z", "+00:00"))
            if stamp.tzinfo is None:
                raise ValueError("Missing timezone on sealed timestamp")
            target_start = datetime.fromisoformat(draw).replace(tzinfo=timezone.utc)
            if stamp.astimezone(timezone.utc) >= target_start:
                raise ValueError(f"Unsealed or late pool: {draw}")
            pools.setdefault(draw, []).append((int(row["rank"]), ticket_of(row), float(row["score"])))
    for draw, entries in pools.items():
        if len(entries) != required_size:
            raise ValueError(f"{draw}: expected {required_size} candidates; got {len(entries)}")
        entries.sort(key=lambda x: x[0])
        if [x[0] for x in entries] != list(range(1, len(entries)+1)):
            raise ValueError(f"{draw}: ranks must be 1..N")
        if len({x[1] for x in entries}) != len(entries):
            raise ValueError(f"{draw}: duplicate tickets")
        if any(not np.isfinite(x[2]) for x in entries):
            raise ValueError(f"{draw}: nonfinite scores")
    return pools

def read_draws(path):
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r["draw_key"]: (tuple(sorted(int(r[f"ball_{i}"]) for i in range(1, 6))),
                              tuple(sorted(int(r[f"star_{i}"]) for i in range(1, 3)))) for r in rows}

def historical_pair_counts(previous_draws):
    counts = Counter()
    for m, _ in previous_draws:
        counts.update(combinations(m, 2))
    return counts

def features(entries, pair_counts):
    scores = np.array([r[2] for r in entries], dtype=np.float64)
    lo, hi = np.quantile(scores, [0.05, 0.95])
    normalized = np.clip((scores-lo) / (hi-lo if hi>lo else 1), 0, 1)
    counts = np.array([min(pair_counts[p] for p in combinations(r[1][0], 2)) for r in entries], dtype=float)
    cutoff = float(np.quantile(list(pair_counts.values()) or [0], 0.15))
    rare = (counts <= cutoff).astype(float)
    return normalized, rare

def metrics(chromosome, entries, scores, rare, max_shared, min_rare_share):
    chosen = [entries[i][1] for i in chromosome]
    n = len(chosen)
    covered = set()
    overlap_violations = 0
    for i, (m, _) in enumerate(chosen):
        covered.update(combinations(m, 2))
        overlap_violations += sum(len(set(m).intersection(chosen[j][0])) > max_shared for j in range(i))
    rare_fraction = float(np.mean(rare[chromosome]))
    # Fitness is computed strictly from presettlement features; no observed prize is referenced.
    objective = (float(np.mean(scores[chromosome])) + 0.50*len(covered)/max(1,10*n)
                 + 0.10*rare_fraction - 3.0*overlap_violations/max(1,n)
                 - 3.0*max(0.0, min_rare_share-rare_fraction))
    return objective, overlap_violations, rare_fraction, len(covered)

def evolve(entries, pair_counts, budget, seed, max_shared=3, min_rare_share=0.0,
           islands=4, population=14, generations=35):
    if len(entries)<budget:
        raise ValueError("Pool smaller than ticket budget")
    rng=np.random.default_rng(seed)
    score, rare=features(entries, pair_counts)
    size=len(entries)
    # Mix model-ranked, rare-pair, and uniform seeds. No target-draw information used.
    rank_seed=np.argsort(-score, kind="stable")[:budget].astype(int)
    rare_seed=np.argsort(-rare-0.01*score, kind="stable")[:budget].astype(int)
    groups=[]
    for island in range(islands):
        members=[rank_seed.copy(), rare_seed.copy()]
        members.extend(rng.choice(size,budget,replace=False) for _ in range(population-2))
        groups.append(members)
    def fitness(x):
        return metrics(x,entries,score,rare,max_shared,min_rare_share)[0]
    for generation in range(generations):
        next_groups=[]
        for members in groups:
            members=sorted(members,key=fitness,reverse=True)
            offspring=[members[0].copy(), members[1].copy()]
            while len(offspring)<population:
                a,b=rng.choice(min(6,len(members)),2,replace=False)
                parent_a,parent_b=members[int(a)],members[int(b)]
                keep=list(dict.fromkeys(list(parent_a[:budget//2])+list(parent_b)))
                if len(keep)<budget:
                    extra=rng.choice(size, min(size,budget*3),replace=False)
                    keep.extend(int(i) for i in extra if int(i) not in keep)
                child=np.asarray(keep[:budget],dtype=int)
                for _ in range(max(1,budget//10)):
                    ix=int(rng.integers(budget))
                    replacement=int(rng.integers(size))
                    if replacement not in child:
                        child[ix]=replacement
                offspring.append(child)
            next_groups.append(offspring)
        groups=next_groups
        if (generation+1)%5==0:
            champions=[max(g,key=fitness).copy() for g in groups]
            for i,g in enumerate(groups):
                g[-1]=champions[(i-1)%len(groups)].copy()
    best=max((member for group in groups for member in group),key=fitness)
    return best, metrics(best,entries,score,rare,max_shared,min_rare_share)

def settle(indices, entries, winning, payouts, price):
    counts=Counter()
    gross=0.0
    for ix in indices:
        ticket=entries[int(ix)][1]
        key=f"{len(set(ticket[0])&set(winning[0]))}+{len(set(ticket[1])&set(winning[1]))}"
        counts[key]+=1
        gross+=float(payouts.get(key,0.0))
    cost=len(indices)*price
    return {"stake":cost,"payout":gross,"net":gross-cost,
            "roi":(gross-cost)/cost if cost else None,
            "tier_counts":dict(counts),
            "best_tier":max(counts,key=lambda x:(int(x.split("+")[0]),int(x.split("+")[1])))}

def evaluate(args):
    pools=read_pools(args.pools,args.pool_size)
    draws=read_draws(args.draws)
    tables=json.loads(Path(args.payouts).read_text(encoding="utf-8"))
    chronological=sorted(draws)
    record=[]
    for draw in sorted(pools):
        if draw not in draws or draw not in tables:
            raise ValueError(f"Missing official winning result or payout schedule: {draw}")
        past=[draws[d] for d in chronological if d<draw]
        if len(past)<args.min_history:
            raise ValueError(f"Insufficient causal history for {draw}")
        entries=pools[draw]
        counts=historical_pair_counts(past)
        settings=[("poi_ga_islands",args.max_shared,args.rare_share),
                  ("poi_ga_relaxed",min(5,args.max_shared+1),0.0)]
        runs=[("poi_g_ranked",np.arange(args.budget),None)]
        random_indices=np.random.default_rng(args.seed+int(draw.replace("-",""))).choice(
            len(entries),args.budget,replace=False)
        runs.append(("uniform_within_pool",random_indices,None))
        for offset,(name,shared,share) in enumerate(settings):
            selection,diag=evolve(entries,counts,args.budget,
                                  args.seed+int(draw.replace("-",""))+offset,
                                  shared,share,args.islands,args.population,args.generations)
            runs.append((name,selection,diag))
        winning=draws[draw]
        winner_in_pool=winning in {e[1] for e in entries}
        for name, selection, diag in runs:
            result=settle(selection,entries,winning,tables[draw],args.ticket_price)
            record.append({"draw_key":draw,"strategy":name,"budget":args.budget,
                           "candidate_pool":len(entries),"winner_in_pool":winner_in_pool,
                           "winner_in_shortlist":winning in {entries[int(i)][1] for i in selection},
                           "optimizer_diagnostics":diag,**result})
    aggregates={}
    for name in sorted({r["strategy"] for r in record}):
        selected=[r for r in record if r["strategy"]==name]
        stake=sum(x["stake"] for x in selected)
        payout=sum(x["payout"] for x in selected)
        aggregates[name]={"draws":len(selected),"stake":stake,"payout":payout,
                          "net":payout-stake,"roi":(payout-stake)/stake,
                          "exact_jackpot_shortlist_hits":sum(x["winner_in_shortlist"] for x in selected),
                          "tier_counts":dict(sum((Counter(x["tier_counts"]) for x in selected),Counter()))}
    result={"parameters":vars(args),"by_draw":record,"aggregates":aggregates,
            "evidence_kind":"historical_walk_forward_paper_settlement",
            "warning":"Selection is based on historical features only. Pool provenance is supplied externally and must be audited."}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pools",required=True)
    p.add_argument("--draws",required=True)
    p.add_argument("--payouts",required=True)
    p.add_argument("--output",default="outputs/poi_islands_result.json")
    p.add_argument("--pool-size",type=int,default=1_000_000)
    p.add_argument("--budget",type=int,default=100)
    p.add_argument("--ticket-price",type=float,required=True)
    p.add_argument("--seed",type=int,default=20261011)
    p.add_argument("--max-shared",type=int,default=3)
    p.add_argument("--rare-share",type=float,default=0.10)
    p.add_argument("--min-history",type=int,default=26)
    p.add_argument("--islands",type=int,default=4)
    p.add_argument("--population",type=int,default=14)
    p.add_argument("--generations",type=int,default=35)
    a=p.parse_args()
    if a.budget<1 or a.pool_size<a.budget or a.ticket_price<=0:
        p.error("Require positive budget, pool size >= budget, ticket price > 0")
    result=evaluate(a)
    print(json.dumps(result["aggregates"],indent=2))

if __name__=="__main__":
    main()
