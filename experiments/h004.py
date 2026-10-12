"""H004: 'number of wins' tiebreak vs coin flip; is it a pairing artifact?"""
from common import sim, cfg, save
from tie_common import tie_run, aggregate

if __name__ == "__main__":
    lines, data = ["Согласие тай-брейка «число побед» с настоящей силой (пары с равными очками), %\n"], {}
    for lab, c in {
        "базовая (ничьи 35%)": cfg(),
        "ничьи 15%": cfg(dmax=0.15),
        "ничьи 55%": cfg(dmax=0.55),
        "без преимущества белых": cfg(white_adv=0.0),
        "случайная жеребьёвка внутри групп": cfg(rule="random"),
        "круговой-подобный: 20 игроков, 15 туров": cfg(N=20, rounds=15),
        "все равны по силе почти (sd 50)": cfg(sd_true=50.0, sd_noise=20.0),
    }.items():
        agg = aggregate(sim(tie_run, c, 2000, seed=4))
        data[lab] = agg
        lines.append(f"  {lab:42s} Wins {100*agg['Wins'][0]:5.1f} ± {100*agg['Wins'][1]:.1f}   "
                     f"(для сравнения BH-C1 {100*agg['BH-C1'][0]:5.1f}, Random {100*agg['Random'][0]:5.1f})")
    save("h004", "\n".join(lines) + "\n", data)
