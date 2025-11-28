import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Football Sniper Ultimate", layout="wide", page_icon="⚽")


# --- 1. FUNKCJE SILNIKA (Backend) ---

@st.cache_data(ttl=3600)
def get_upcoming_fixtures(league_name):
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        # Kodowanie utf-8-sig usuwa BOM
        df = pd.read_csv(url, encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = df.columns.str.strip()

        # Auto-naprawa nazwy kolumny Div
        if 'Div' not in df.columns:
            for col in df.columns:
                if 'Div' in col or 'Division' in col:
                    df = df.rename(columns={col: 'Div'})
                    break
        if 'Div' not in df.columns: return pd.DataFrame()

        league_map = {
            "Premier League": "E0", "Championship": "E1", "League One": "E2", "League Two": "E3",
            "La Liga": "SP1", "La Liga 2": "SP2", "Bundesliga": "D1", "Bundesliga 2": "D2",
            "Serie A": "I1", "Serie B": "I2", "Ligue 1": "F1", "Ligue 2": "F2",
            "Eredivisie": "N1", "Jupiler League": "B1", "Liga Portugal": "P1",
            "Super Lig": "T1", "Greece Super League": "G1", "Scottish Premiership": "SC0"
        }
        div_code = league_map.get(league_name)
        league_fixtures = df[df['Div'] == div_code].copy()
        league_fixtures['Date'] = pd.to_datetime(league_fixtures['Date'], dayfirst=True, errors='coerce')

        today = pd.Timestamp.now().normalize()
        future_games = league_fixtures[league_fixtures['Date'] >= today]
        return future_games.sort_values(['Date', 'Time'])
    except:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_and_prep_data(league_name):
    # --- BEZPIECZNE POŁĄCZENIE ---
    try:
        # Streamlit sam znajdzie plik secrets.toml (lokalnie) lub Secrets (w chmurze)
        db_url = st.secrets["DB_URL"]
        engine = create_engine(db_url)
    except Exception as e:
        st.error("❌ Błąd: Nie znaleziono sekretów bazy danych!")
        st.info("Upewnij się, że masz plik .streamlit/secrets.toml (lokalnie) lub skonfigurowane Secrets (w chmurze).")
        st.stop()

    query = f"SELECT * FROM matches WHERE league = '{league_name}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    # Mapowanie nazw kolumn z bazy na nazwy używane w kodzie (skróty)
    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC',
        'home_shots': 'HS', 'away_shots': 'AS',
        'home_yellow': 'HY', 'away_yellow': 'AY',
        'home_fouls': 'HF', 'away_fouls': 'AF'
    })
    return df


def add_rolling_features(df, window=5):
    data = df.copy()

    # Uzupełnianie zerami brakujących kolumn (dla starszych danych)
    for col in ['HS', 'AS', 'HY', 'AY', 'HF', 'AF']:
        if col not in data.columns: data[col] = 0

    # 1. Home Stats
    home_df = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HS', 'AS', 'HY', 'HF']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GS', 'FTAG': 'GC',
                 'HC': 'CorW', 'AC': 'CorL',
                 'HS': 'ShotsW', 'AS': 'ShotsL',
                 'HY': 'Cards', 'HF': 'Fouls'}
    )
    home_df['Pts'] = np.where(home_df['FTR'] == 'H', 3, np.where(home_df['FTR'] == 'D', 1, 0))
    home_df['IsHome'] = 1

    # 2. Away Stats
    away_df = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AC', 'HC', 'AS', 'HS', 'AY', 'AF']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GS', 'FTHG': 'GC',
                 'AC': 'CorW', 'HC': 'CorL',
                 'AS': 'ShotsW', 'HS': 'ShotsL',
                 'AY': 'Cards', 'AF': 'Fouls'}
    )
    away_df['Pts'] = np.where(away_df['FTR'] == 'A', 3, np.where(away_df['FTR'] == 'D', 1, 0))
    away_df['IsHome'] = 0

    # 3. Rolling
    team_stats = pd.concat([home_df, away_df]).sort_values('Date')

    features = team_stats.groupby('Team')[['GS', 'GC', 'Pts', 'CorW', 'ShotsW', 'Cards', 'Fouls']].transform(
        lambda x: x.rolling(window, min_periods=3).mean().shift(1)
    )

    team_stats[['Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor', 'Form_Shots', 'Form_Cards', 'Form_Fouls']] = features

    # 4. Merge
    cols_to_keep = ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor', 'Form_Shots', 'Form_Cards',
                    'Form_Fouls']

    h_stats = team_stats[team_stats['IsHome'] == 1][cols_to_keep].rename(
        columns={'Team': 'HomeTeam',
                 'Form_Att': 'Home_Att', 'Form_Def': 'Home_Def',
                 'Form_Pts': 'Home_Form', 'Form_Cor': 'Home_Corners_Avg',
                 'Form_Shots': 'Home_Shots_Avg', 'Form_Cards': 'Home_Cards_Avg', 'Form_Fouls': 'Home_Fouls_Avg'})

    a_stats = team_stats[team_stats['IsHome'] == 0][cols_to_keep].rename(
        columns={'Team': 'AwayTeam',
                 'Form_Att': 'Away_Att', 'Form_Def': 'Away_Def',
                 'Form_Pts': 'Away_Form', 'Form_Cor': 'Away_Corners_Avg',
                 'Form_Shots': 'Away_Shots_Avg', 'Form_Cards': 'Away_Cards_Avg', 'Form_Fouls': 'Away_Fouls_Avg'})

    data = pd.merge(data, h_stats, on=['Date', 'HomeTeam'], how='left')
    data = pd.merge(data, a_stats, on=['Date', 'AwayTeam'], how='left')
    data['OddsDiff'] = (1 / data['B365H']) - (1 / data['B365A'])

    return data.dropna()


@st.cache_resource
def train_model(df, prediction_type="1X2"):
    predictors = [
        'OddsDiff',
        'Home_Att', 'Away_Att',
        'Home_Def', 'Away_Def',
        'Home_Form', 'Away_Form',
        'Home_Corners_Avg', 'Away_Corners_Avg',
        'Home_Shots_Avg', 'Away_Shots_Avg',
        'Home_Cards_Avg', 'Away_Cards_Avg',
        'Home_Fouls_Avg', 'Away_Fouls_Avg'
    ]

    best_params = {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'subsample': 0.7,
                   'colsample_bytree': 0.8, 'n_jobs': -1}

    if prediction_type == "1X2":
        predictors += ['B365H', 'B365A', 'B365D']
        df['Target'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, **best_params)
        model.fit(df[predictors], df['Target'])
    else:
        predictors += ['B365_O25', 'B365_U25']
        target = np.where((df['FTHG'] + df['FTAG']) > 2.5, 1, 0)
        model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
        model.fit(df[predictors], target)
    return model, predictors


def calculate_kelly(prob, odds, bankroll, fraction):
    if prob * odds <= 1: return 0.0
    b = odds - 1
    return bankroll * max(((b * prob - (1 - prob)) / b), 0) * fraction


def plot_score_heatmap(home_exp, away_exp):
    probs = np.zeros((5, 5))
    for h in range(5):
        for a in range(5):
            probs[a, h] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(probs, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel("Gole Gospodarza")
    ax.set_ylabel("Gole Gościa")
    ax.set_title("Dokładny Wynik (%)")
    return fig


# --- 2. INTERFEJS GŁÓWNY ---

st.title("⚽ AI Football Sniper Ultimate")

# --- PANEL BOCZNY ---
st.sidebar.header("Ustawienia")

league_config = {
    # GRUPA "ELITA" (Graj Odważnie)
    "Serie B": {"roi_1x2": 35.46, "roi_ou": 6.29, "recom": "🔥 GRAJ WSZYSTKO"},
    "Bundesliga": {"roi_1x2": 23.58, "roi_ou": 5.63, "recom": "🔥 GRAJ WSZYSTKO (Nowy Król!)"},
    "Greece Super League": {"roi_1x2": 15.68, "roi_ou": 34.88, "recom": "🔥 GRAJ WSZYSTKO"},
    "Championship": {"roi_1x2": 12.55, "roi_ou": 14.51, "recom": "🔥 GRAJ WSZYSTKO"},

    # GRUPA "SOLIDNE" (Wybieraj konkretne rynki)
    "Serie A": {"roi_1x2": 13.97, "roi_ou": -4.63, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "La Liga 2": {"roi_1x2": 16.89, "roi_ou": 4.25, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "Scottish Premiership": {"roi_1x2": 9.43, "roi_ou": -44.03, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "Jupiler League": {"roi_1x2": -31.89, "roi_ou": 23.39, "recom": "✅ GRAJ GOLE (O/U)"},
    "Super Lig": {"roi_1x2": -2.77, "roi_ou": 8.82, "recom": "✅ GRAJ GOLE (O/U)"},

    # GRUPA "CZERWONA" (Ignoruj)
    "Premier League": {"roi_1x2": -16.94, "roi_ou": -29.05, "recom": "⛔ UNIKAJ (Strata pieniędzy)"},
    "La Liga": {"roi_1x2": -17.99, "roi_ou": -13.01, "recom": "⛔ UNIKAJ (Za trudna technicznie)"},
    "Ligue 1": {"roi_1x2": -52.57, "roi_ou": -27.77, "recom": "⛔ UNIKAJ"},
    "Ligue 2": {"roi_1x2": -36.25, "roi_ou": -46.01, "recom": "⛔ UNIKAJ"},
    "Eredivisie": {"roi_1x2": -11.15, "roi_ou": -30.33, "recom": "⛔ UNIKAJ"},
    "Liga Portugal": {"roi_1x2": -9.40, "roi_ou": -36.63, "recom": "⛔ UNIKAJ"},
    "League One": {"roi_1x2": -6.32, "roi_ou": -29.06, "recom": "⛔ UNIKAJ"},
    "League Two": {"roi_1x2": -15.83, "roi_ou": -6.65, "recom": "⛔ UNIKAJ"},
    "Bundesliga 2": {"roi_1x2": -0.02, "roi_ou": -33.51, "recom": "⛔ UNIKAJ"},
}

selected_league = st.sidebar.selectbox("Wybierz Ligę", sorted(list(league_config.keys())))
bet_type = st.sidebar.radio("Strategia", ["Zwycięzca (1X2)", "Gole (Over/Under 2.5)"])
bankroll = st.sidebar.number_input("Bankroll (PLN)", 1000)
kelly_frac = st.sidebar.slider("Kelly %", 0.05, 0.2, 0.1)

cfg = league_config.get(selected_league, {"roi_1x2": 0, "roi_ou": 0, "recom": "?"})
st.sidebar.info(f"{cfg['recom']}")
c1, c2 = st.sidebar.columns(2)
c1.metric("ROI 1X2", f"{cfg['roi_1x2']}%")
c2.metric("ROI O/U", f"{cfg['roi_ou']}%")

# ŁADOWANIE DANYCH
with st.spinner("Ładowanie AI..."):
    raw_data = load_and_prep_data(selected_league)
    if raw_data.empty: st.error("Brak danych"); st.stop()
    processed_data = add_rolling_features(raw_data)
    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, train_mode)

# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs(["🔥 Skaner Value", "🔍 Analiza Meczu", "📜 Historia"])

# --- TAB 1: SKANER ---
with tab1:
    st.header(f"Skaner: {selected_league}")
    fixtures = get_upcoming_fixtures(selected_league)

    if fixtures.empty:
        st.info("Brak meczów. Sprawdź w piątek!")
    else:
        value_bets = []
        progress = st.progress(0)
        total_games = len(fixtures)

        # Pętla z enumerate (naprawiony błąd progress bar)
        for i, (index, row) in enumerate(fixtures.iterrows()):
            progress.progress(min((i + 1) / total_games, 1.0))

            h_stats = processed_data[processed_data['HomeTeam'] == row['HomeTeam']]
            a_stats = processed_data[processed_data['AwayTeam'] == row['AwayTeam']]
            if h_stats.empty or a_stats.empty: continue

            h_stat = h_stats.iloc[-1]
            a_stat = a_stats.iloc[-1]

            # INPUT Z PEŁNYMI STATYSTYKAMI
            match_input = pd.DataFrame([{
                'Home_Att': h_stat['Home_Att'], 'Away_Att': a_stat['Away_Att'],
                'Home_Def': h_stat['Home_Def'], 'Away_Def': a_stat['Away_Def'],
                'Home_Form': h_stat['Home_Form'], 'Away_Form': a_stat['Away_Form'],
                'Home_Corners_Avg': h_stat['Home_Corners_Avg'], 'Away_Corners_Avg': a_stat['Away_Corners_Avg'],
                'Home_Shots_Avg': h_stat['Home_Shots_Avg'], 'Away_Shots_Avg': a_stat['Away_Shots_Avg'],
                'Home_Cards_Avg': h_stat['Home_Cards_Avg'], 'Away_Cards_Avg': a_stat['Away_Cards_Avg'],
                'Home_Fouls_Avg': h_stat['Home_Fouls_Avg'], 'Away_Fouls_Avg': a_stat['Away_Fouls_Avg']
            }])

            if bet_type == "Zwycięzca (1X2)":
                o1 = row.get('B365H');
                ox = row.get('B365D');
                o2 = row.get('B365A')
                if pd.isna(o1) or pd.isna(ox) or pd.isna(o2): continue

                match_input['OddsDiff'] = (1 / o1) - (1 / o2)
                match_input['B365H'] = o1;
                match_input['B365D'] = ox;
                match_input['B365A'] = o2
                match_input = match_input[features]  # Naprawa kolejności kolumn

                probs = model.predict_proba(match_input)[0]
                for outcome, prob, odd, name in [(2, probs[2], o1, row['HomeTeam']),
                                                 (0, probs[0], o2, row['AwayTeam'])]:
                    val = (prob * odd) - 1
                    if val > 0.05:
                        value_bets.append(
                            {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                             "Typ": name, "Kurs": odd, "Szansa AI": f"{prob * 100:.1f}%", "Value": f"{val * 100:.1f}%",
                             "Rekomendacja": "🔥 GRAJ"})
            else:
                oo = row.get('B365>2.5');
                ou = row.get('B365<2.5')
                if pd.isna(oo) or pd.isna(ou): continue
                match_input['OddsDiff'] = (1 / oo) - (1 / ou)
                match_input['B365_O25'] = oo;
                match_input['B365_U25'] = ou
                match_input = match_input[features]  # Naprawa kolejności kolumn

                p_over = model.predict_proba(match_input)[0][1]
                p_under = 1.0 - p_over

                val_o = (p_over * oo) - 1
                if val_o > 0.05: value_bets.append(
                    {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                     "Typ": "OVER 2.5", "Kurs": oo, "Szansa AI": f"{p_over * 100:.1f}%", "Value": f"{val_o * 100:.1f}%",
                     "Rekomendacja": "🔥 GRAJ"})
                val_u = (p_under * ou) - 1
                if val_u > 0.05: value_bets.append(
                    {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                     "Typ": "UNDER 2.5", "Kurs": ou, "Szansa AI": f"{p_under * 100:.1f}%",
                     "Value": f"{val_u * 100:.1f}%", "Rekomendacja": "🔥 GRAJ"})

        progress.empty()
        if value_bets:
            st.dataframe(pd.DataFrame(value_bets), use_container_width=True)
        else:
            st.warning("Brak Value > 5%")

# --- TAB 2: ANALIZA MANUALNA ---
with tab2:
    match_opts = ["Wybierz ręcznie..."]
    match_map = {}
    if not fixtures.empty:
        for idx, row in fixtures.iterrows():
            lbl = f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} vs {row['AwayTeam']}"
            match_opts.append(lbl)
            match_map[lbl] = row

    sel_fix = st.selectbox("Terminarz", match_opts)
    def_h, def_a = 0, 1
    d_o1, d_ox, d_o2 = 2.0, 3.2, 3.5
    d_oo, d_ou = 1.9, 1.9

    if sel_fix != "Wybierz ręcznie...":
        md = match_map[sel_fix]
        teams = sorted(raw_data['HomeTeam'].unique())
        try:
            def_h = teams.index(md['HomeTeam']);
            def_a = teams.index(md['AwayTeam'])
            if pd.notnull(md.get('B365H')): d_o1 = float(md['B365H'])
            if pd.notnull(md.get('B365D')): d_ox = float(md['B365D'])
            if pd.notnull(md.get('B365A')): d_o2 = float(md['B365A'])
            if pd.notnull(md.get('B365>2.5')): d_oo = float(md['B365>2.5'])
            if pd.notnull(md.get('B365<2.5')): d_ou = float(md['B365<2.5'])
        except:
            pass

    c1, c2 = st.columns(2)
    teams = sorted(raw_data['HomeTeam'].unique())
    home_team = c1.selectbox("Gospodarz", teams, index=def_h)
    away_team = c2.selectbox("Gość", teams, index=def_a)

    k1, k2, k3 = st.columns(3)
    if bet_type == "Zwycięzca (1X2)":
        o1 = k1.number_input("1", d_o1);
        ox = k2.number_input("X", d_ox);
        o2 = k3.number_input("2", d_o2)
    else:
        oo = k1.number_input("Over", d_oo);
        ou = k2.number_input("Under", d_ou)
        o1, o2, ox = 0, 0, 0

    if st.button("ANALIZA", type="primary"):
        h_stat = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stat = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        input_data = pd.DataFrame([{
            'Home_Att': h_stat['Home_Att'], 'Away_Att': a_stat['Away_Att'],
            'Home_Def': h_stat['Home_Def'], 'Away_Def': a_stat['Away_Def'],
            'Home_Form': h_stat['Home_Form'], 'Away_Form': a_stat['Away_Form'],
            'Home_Corners_Avg': h_stat['Home_Corners_Avg'], 'Away_Corners_Avg': a_stat['Away_Corners_Avg'],
            'Home_Shots_Avg': h_stat['Home_Shots_Avg'], 'Away_Shots_Avg': a_stat['Away_Shots_Avg'],
            'Home_Cards_Avg': h_stat['Home_Cards_Avg'], 'Away_Cards_Avg': a_stat['Away_Cards_Avg'],
            'Home_Fouls_Avg': h_stat['Home_Fouls_Avg'], 'Away_Fouls_Avg': a_stat['Away_Fouls_Avg']
        }])

        if bet_type == "Zwycięzca (1X2)":
            input_data['OddsDiff'] = (1 / o1) - (1 / o2)
            input_data['B365H'] = o1;
            input_data['B365D'] = ox;
            input_data['B365A'] = o2
            input_data = input_data[features]  # Sort
            probs = model.predict_proba(input_data)[0]
            outcomes = [("1", probs[2], o1), ("X", probs[1], ox), ("2", probs[0], o2)]
        else:
            input_data['OddsDiff'] = (1 / oo) - (1 / ou)
            input_data['B365_O25'] = oo;
            input_data['B365_U25'] = ou
            input_data = input_data[features]  # Sort
            p_over = model.predict_proba(input_data)[0][1]
            outcomes = [("Over", p_over, oo), ("Under", 1 - p_over, ou)]

        cols = st.columns(len(outcomes))
        for i, (lbl, p, o) in enumerate(outcomes):
            with cols[i]:
                val = (p * o) - 1
                st.metric(lbl, f"{p * 100:.1f}%", f"Val: {val * 100:.1f}%", delta_color="normal" if val > 0 else "off")
                if val > 0.05: st.success(f"Stawka: {calculate_kelly(p, o, bankroll, kelly_frac):.2f}")

        st.divider()
        xg_h = (h_stat['Home_Att'] + a_stat['Away_Def']) / 2
        xg_a = (a_stat['Away_Att'] + h_stat['Home_Def']) / 2
        c1, c2 = st.columns([2, 1])
        c1.pyplot(plot_score_heatmap(xg_h, xg_a))
        c2.info(f"xG: {xg_h:.2f} - {xg_a:.2f}")

# --- TAB 3: HISTORIA ---
with tab3:
    st.header("Weryfikacja Modelu")
    st.markdown("Ostatnie 20 meczów i decyzje, jakie podjąłby model (Value > 10%).")

    # Sortujemy od najnowszych
    history = processed_data.sort_values("Date", ascending=False).head(20).copy()
    rows = []

    for idx, row in history.iterrows():
        # Rekonstrukcja cech do modelu
        input_dict = {col: row[col] for col in features if col in row}

        # Jeśli brakuje OddsDiff (bo to cecha liczona dynamicznie), dodajemy ją
        if 'OddsDiff' not in input_dict:
            input_dict['OddsDiff'] = (1 / row['B365H']) - (1 / row['B365A'])

        clean_input = pd.DataFrame([input_dict])
        clean_input = clean_input[features]  # Sortowanie kolumn zgodnie z modelem

        pick = "-"  # Co wybrał model
        res = "⚪"  # Czy weszło (kolor)
        played_odd = 0.0  # Po jakim kursie

        if bet_type == "Zwycięzca (1X2)":
            probs = model.predict_proba(clean_input)[0]

            # Sprawdzamy Home
            if (probs[2] * row['B365H']) - 1 > 0.1:
                pick = "HOME"
                played_odd = row['B365H']
                res = "🟢 WIN" if row['FTR'] == 'H' else "🔴 LOSS"

            # Sprawdzamy Away
            elif (probs[0] * row['B365A']) - 1 > 0.1:
                pick = "AWAY"
                played_odd = row['B365A']
                res = "🟢 WIN" if row['FTR'] == 'A' else "🔴 LOSS"

            # Sprawdzamy Draw (opcjonalnie, jeśli model widzi value w remisie)
            elif (probs[1] * row['B365D']) - 1 > 0.1:
                pick = "DRAW"
                played_odd = row['B365D']
                res = "🟢 WIN" if row['FTR'] == 'D' else "🔴 LOSS"

        else:  # Over/Under
            p_over = model.predict_proba(clean_input)[0][1]
            p_under = 1.0 - p_over

            total_goals = row['FTHG'] + row['FTAG']

            # Sprawdzamy Over
            if (p_over * row['B365_O25']) - 1 > 0.1:
                pick = "OVER"
                played_odd = row['B365_O25']
                res = "🟢 WIN" if total_goals > 2.5 else "🔴 LOSS"

            # Sprawdzamy Under
            elif (p_under * row['B365_U25']) - 1 > 0.1:
                pick = "UNDER"
                played_odd = row['B365_U25']
                res = "🟢 WIN" if total_goals < 2.5 else "🔴 LOSS"

        # Formatowanie kursu do wyświetlenia
        odd_display = f"{played_odd:.2f}" if pick != "-" else "-"

        rows.append({
            "Data": row['Date'].strftime('%d.%m'),
            "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
            "Wynik": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
            "Typ AI": pick,
            "Kurs": odd_display,  # <--- Tego brakowało!
            "Rozliczenie": res
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)