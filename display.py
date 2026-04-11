from datetime import date as _date
from pathlib import Path
from models import DayRecommendation, TripContext
import os
import subprocess
import sys
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _fmt_date(d: str) -> str:
    try:
        return _date.fromisoformat(d).strftime("%a %d %b")
    except Exception:
        return d


def _clothing_narrative(rec) -> str:
    """
    Turn a day's clothing list into a readable paragraph.
    Groups items into base layers, outer layers, footwear, and purpose items,
    then weaves them into a flowing sentence or two.
    """
    BASE_LAYER   = {"T-shirt or short sleeves", "Lightweight breathable clothing",
                    "Long-sleeve shirt", "Thermal underlayer", "Warm sweater or fleece",
                    "Shorts or light trousers", "Jeans or trousers",
                    "Casual wear", "Smart casual outfit", "Business attire"}
    OUTER_LAYER  = {"Heavy winter coat", "Light jacket or fleece", "Waterproof jacket",
                    "Windproof jacket", "Smart jacket"}
    FOOTWEAR     = {"Insulated boots", "Waterproof snow boots", "Comfortable walking shoes",
                    "Formal shoes"}
    ACCESSORIES  = {"Gloves and scarf", "Thermal socks"}

    clothing = rec.clothing
    if not clothing:
        return f"On {_fmt_date(rec.date)}, the weather is {rec.summary}. No specific clothing changes needed for the day."

    base    = [c for c in clothing if c in BASE_LAYER]
    outer   = [c for c in clothing if c in OUTER_LAYER]
    shoes   = [c for c in clothing if c in FOOTWEAR]
    acc     = [c for c in clothing if c in ACCESSORIES]
    other   = [c for c in clothing if c not in BASE_LAYER | OUTER_LAYER | FOOTWEAR | ACCESSORIES]

    parts = []
    if base:
        parts.append("For your base layer, go with " + _join_items(base))
    if outer:
        parts.append("bring " + _join_items(outer) + " for the outer layer")
    if shoes:
        parts.append(_join_items(shoes) + " will be the right footwear choice")
    if acc:
        parts.append("add " + _join_items(acc) + " to stay comfortable")
    if other:
        parts.append("also consider " + _join_items(other))

    sentence = ". ".join(p[0].upper() + p[1:] for p in parts) + "."
    intro = f"On {_fmt_date(rec.date)}, expect {rec.summary}."
    return f"{intro} {sentence}"


def _join_items(items: list) -> str:
    if len(items) == 1:
        return items[0].lower()
    return ", ".join(i.lower() for i in items[:-1]) + f" and {items[-1].lower()}"


def _alert_color(alerts: list) -> str:
    if not alerts:
        return "green"
    text = " ".join(alerts).lower()
    if any(w in text for w in ("heavy", "violent", "strong", "thunderstorm")):
        return "red"
    return "yellow"


# ── Matplotlib forecast chart ─────────────────────────────────────────────────

def plot_forecast(forecasts: list, context: TripContext,
                  start_date: str, end_date: str) -> Path:
    """
    Generate a 3-panel chart:
      1. Temperature band (min/max fill) + average line
      2. Precipitation bar chart
      3. UV index line with risk-level shading
    Saves to <city>_<start_date>_forecast.png and returns the path.
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np

    dates    = [_date.fromisoformat(f.date) for f in forecasts]
    temp_min = np.array([f.temp_min for f in forecasts])
    temp_max = np.array([f.temp_max for f in forecasts])
    temp_avg = (temp_min + temp_max) / 2
    precip   = np.array([f.precipitation_mm for f in forecasts])
    uv       = np.array([f.uv_index_max for f in forecasts])

    method_label = (
        f"Historical Prediction (past climate avg + trend)"
    )
    title = (f"{context.city}, {context.country}  |  "
             f"{start_date} → {end_date}  |  {method_label}")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)

    # ── Panel 1: Temperature ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.fill_between(dates, temp_min, temp_max,
                     alpha=0.25, color="tomato", label="Min–Max range")
    ax1.plot(dates, temp_avg, color="crimson",
             linewidth=2, marker="o", markersize=4, label="Avg temp")
    ax1.plot(dates, temp_min, color="steelblue",
             linewidth=1, linestyle="--", alpha=0.6, label="Min")
    ax1.plot(dates, temp_max, color="darkorange",
             linewidth=1, linestyle="--", alpha=0.6, label="Max")
    ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # ── Panel 2: Precipitation ────────────────────────────────────────────────
    ax2 = axes[1]
    colors = ["#d32f2f" if p > 20 else "#1976d2" if p > 1 else "#90caf9"
              for p in precip]
    ax2.bar(dates, precip, color=colors, alpha=0.8, width=0.6, label="Precipitation")
    ax2.axhline(1,  color="gray",  linewidth=0.8, linestyle="--", alpha=0.5, label="1mm threshold")
    ax2.axhline(20, color="red",   linewidth=0.8, linestyle="--", alpha=0.5, label="Heavy rain (20mm)")
    ax2.set_ylabel("Precipitation (mm)")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: UV Index ─────────────────────────────────────────────────────
    ax3 = axes[2]
    # Risk-level background bands
    ax3.axhspan(0,  3,  alpha=0.07, color="green")
    ax3.axhspan(3,  6,  alpha=0.07, color="yellow")
    ax3.axhspan(6,  8,  alpha=0.07, color="orange")
    ax3.axhspan(8,  16, alpha=0.07, color="red")
    for level, label, color in [(3, "Moderate (3)", "gold"),
                                 (6, "High (6)", "orange"),
                                 (8, "Very High (8)", "red")]:
        ax3.axhline(level, color=color, linewidth=0.8, linestyle="--", alpha=0.6, label=label)
    ax3.plot(dates, uv, color="darkorange",
             linewidth=2, marker="s", markersize=4, label="UV Index")
    ax3.set_ylabel("UV Index")
    ax3.set_ylim(bottom=0)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(True, alpha=0.3)

    # ── X-axis formatting ─────────────────────────────────────────────────────
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax3.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates) // 10)))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    city_slug = context.city.lower().replace(" ", "_").replace(",", "")
    out_path = Path(__file__).parent / f"{city_slug}_{start_date}_forecast.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return out_path


# ── Terminal display ──────────────────────────────────────────────────────────

def display_rich(context, start_date, end_date, recommendations, trip_packing, n_years=None):
    console = Console()

    method_tag = f"[yellow]Historical Prediction ({n_years}yr avg + trend)[/]"

    header = (
        f"[bold]Trip:[/] {context.city}, {context.country}  |  "
        f"[bold]Dates:[/] {_fmt_date(start_date)} → {_fmt_date(end_date)}  |  "
        f"[bold]Purpose:[/] {context.purpose.title()}  |  "
        f"[bold]Method:[/] {method_tag}"
    )
    console.print(Panel(header, title="[bold cyan]Travel Weather Advisor[/]", border_style="cyan"))

    table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
    table.add_column("Date",     style="cyan",  no_wrap=True)
    table.add_column("Weather",  style="white")
    table.add_column("Clothing", style="white")
    table.add_column("Alerts",   style="white")

    for rec in recommendations:
        alert_color  = _alert_color(rec.alerts)
        alerts_text  = "\n".join(rec.alerts) if rec.alerts else "[green]All good[/]"
        clothing_text = "\n".join(f"• {c}" for c in rec.clothing[:4])
        if len(rec.clothing) > 4:
            clothing_text += f"\n  [dim]+ {len(rec.clothing) - 4} more[/]"
        table.add_row(
            _fmt_date(rec.date), rec.summary, clothing_text,
            f"[{alert_color}]{alerts_text}[/]",
        )
    console.print(table)

    clothing_lines = "\n".join(f"  • {c}" for c in trip_packing["clothing"])
    packing_lines  = "\n".join(f"  • {p}" for p in trip_packing["packing"])
    console.print(Panel(
        f"[bold]Clothing:[/]\n{clothing_lines}\n\n[bold]Gear & Essentials:[/]\n{packing_lines}",
        title="[bold green]Master Packing List[/]", border_style="green",
    ))

    tips = []
    for rec in recommendations:
        if rec.alerts:
            tips.append(f"[cyan]{_fmt_date(rec.date)}[/]: " + "  " + "\n  ".join(rec.alerts))
    if tips:
        console.print(Panel("\n".join(tips), title="[bold yellow]Day-by-Day Alerts[/]", border_style="yellow"))

    narratives = "\n\n".join(_clothing_narrative(rec) for rec in recommendations)

    #call claude to get the suggestions
    if False:
        proc = subprocess.run(
        [
                                "Claude",
                                "based on the trip day weather forcast and cloth suggestions:"+narratives+", this folder have 3 pictures, please let me know which I should take during the trip? not too long explain   ",#"read claude.md",#
                                "--dangerously-skip-permissions",
                                "--print",
                                "--verbose",
                                "--output-format",
                                "text"#"stream-json",
        ],
        cwd="/Users/zhangs/Documents/NUS-MTEC/GP-2 (existingForcast+history)/img",#task-schedule-processor",
        check=False,
        capture_output=True,
        text=True,
        )

        out = (proc.stdout or "").strip()
        print("out:", out)
        err = (proc.stderr or "").strip()
        print("err:", err)
        msg = out or err or f"Claude playbook finished (exit code {proc.returncode})."

    else:

        from google import genai
        from google.genai import types
        import PIL.Image                                                                   
        
                                
        #img = PIL.Image.open("img/IMG_1.jpg")
        images = [PIL.Image.open(f"img/IMG_{i}.jpg") for i in range(3)]

        response = ""#client.models.generate_content(                                         
            #model="gemini-2.5-flash",
            #contents=["based on the trip day weather forcast and cloth suggestions:"+narratives+", please let me know what is this picture weight and volume ,should I take it during the trip? not too long explain", *images]  # PIL Image works directly             
        #)                                                                                  
        #print(response.text)
        msg ="test"#response.text
    



    console.print(Panel(msg, title="[bold blue] Suggestions[/]", border_style="blue"))


def display_plain(context, start_date, end_date, recommendations, trip_packing, n_years=None):
    sep = "-" * 60
    print(sep)
    print("Travel Weather Advisor")
    print(f"Trip   : {context.city}, {context.country}")
    print(f"Dates  : {start_date} to {end_date}")
    print(f"Purpose: {context.purpose.title()}")
    print(sep)
    for rec in recommendations:
        print(f"\n{_fmt_date(rec.date)} — {rec.summary}")
        for c in rec.clothing:
            print(f"    • {c}")
        for p in rec.packing:
            print(f"    * {p}")
        for a in rec.alerts:
            print(f"    ! {a}")
    print(f"\n{sep}\nMASTER PACKING LIST\n{sep}")
    for c in trip_packing["clothing"]:
        print(f"  • {c}")
    for p in trip_packing["packing"]:
        print(f"  * {p}")
    print(sep)
    print(f"\n{sep}\nDAILY CLOTHING SUGGESTIONS\n{sep}")
    for rec in recommendations:
        print()
        print(_clothing_narrative(rec))
    print(sep)


def display(context, start_date, end_date, recommendations, trip_packing, n_years=None):
    if RICH_AVAILABLE:
        display_rich(context, start_date, end_date, recommendations, trip_packing, n_years)
    else:
        display_plain(context, start_date, end_date, recommendations, trip_packing, n_years)
