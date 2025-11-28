import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Football Predictor 2.0", layout="wide")


# --- 1. FUNKCJE SILNIKA ---

@st.cache_data(ttl=3600)
def get_upcoming_fixtures(league_name):
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        # Kodowanie utf-8-sig usuwa "krzaczki" (BOM)
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
            "La Liga": "SP1", "La Liga 2": "SP2",
            "Bundesliga": "D1", "Bundesliga 2": "D2",
            "Serie A": "I1", "Serie B": "I2",
            "Ligue 1": "F1", "Ligue 2": "F2",
            "Eredivisie": "N1", "Jupiler League": "B1", "Liga Portugal": "P1",
            "Super Lig": "T1", "Greece Super League": "G1", "Scottish Premiership": "SC0"
        }

        div_code = league_map.get(league_name)
        league_fixtures = df[df['Div'] == div_code].copy()

        league_fixtures['Date'] = pd.to_datetime(league_fixtures['Date'], dayfirst=True, errors='coerce')

        # Filtr: Tylko mecze od dzisiaj w górę
        today = pd.Timestamp.now().normalize()
        future_games = league_fixtures[league_fixtures['Date'] >= today]

        return future_games.sort_values(['Date', 'Time'])

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_and_prep_data(league_name):
    # HYBRYDOWE POŁĄCZENIE (Działa lokalnie i w chmurze)
    try:
        # Próba 1: Chmura (Streamlit Secrets)
        db_url = st.secrets["DB_URL"]
        engine = create_engine(db_url)
    except:
        # Próba 2: Lokalnie (Twój komputer)
        DB_USER = 'postgres'
        DB_PASS = 'EAtocepy12!'
        DB_HOST = 'localhost'
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/postgres')

    query = f"SELECT * FROM matches WHERE league = '{league_name}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC'
    })
    return df


def add_rolling_features(df, window=5):
    data = df.copy()

    # 1. Home
    home_df = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GS', 'FTAG': 'GC', 'HC': 'CorW', 'AC': 'CorL'})
    home_df['Pts'] = np.where(home_df['FTR'] == 'H', 3, np.where(home_df['FTR'] == 'D', 1, 0))
    home_df['IsHome'] = 1

    # 2. Away
    away_df = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AC', 'HC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GS', 'FTHG': 'GC', 'AC': 'CorW', 'HC': 'CorL'})
    away_df['Pts'] = np.where(away_df['FTR'] == 'A', 3, np.where(away_df['FTR'] == 'D', 1, 0))
    away_df['IsHome'] = 0

    # 3. Rolling
    team_stats = pd.concat([home_df, away_df]).sort_values('Date')
    features = team_stats.groupby('Team')[['GS', 'GC', 'Pts', 'CorW']].transform(
        lambda x: x.rolling(window, min_periods=3).mean().shift(1)
    )

    team_stats[['Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']] = features

    # 4. Merge
    h_stats = team_stats[team_stats['IsHome'] == 1][
        ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']].rename(
        columns={'Team': 'HomeTeam', 'Form_Att': 'Home_Att', 'Form_Def': 'Home_Def', 'Form_Pts': 'Home_Form',
                 'Form_Cor': 'Home_Corners_Avg'})
    a_stats = team_stats[team_stats['IsHome'] == 0][
        ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor']].rename(
        columns={'Team': 'AwayTeam', 'Form_Att': 'Away_Att', 'Form_Def': 'Away_Def', 'Form_Pts': 'Away_Form',
                 'Form_Cor': 'Away_Corners_Avg'})

    data = pd.merge(data, h_stats, on=['Date', 'HomeTeam'], how='left')
    data = pd.merge(data, a_stats, on=['Date', 'AwayTeam'], how='left')
    data['OddsDiff'] = (1 / data['B365H']) - (1 / data['B365A'])

    return data.dropna()


@st.cache_resource
def train_model(df, prediction_type="1X2"):
    predictors = ['OddsDiff', 'Home_Att', 'Away_Att', 'Home_Def', 'Away_Def', 'Home_Form', 'Away_Form',
                  'Home_Corners_Avg', 'Away_Corners_Avg']

    # Parametry z Optymalizacji
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
    full = (b * prob - (1 - prob)) / b
    return bankroll * max(full, 0) * fraction


def plot_score_heatmap(home_exp, away_exp):
    probs = np.zeros((5, 5))
    for h in range(5):
        for a in range(5):
            probs[a, h] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp) * 100

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(probs, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel("Gole Gospodarza")
    ax.set_ylabel("Gole Gościa")
    ax.set_title("Prawdopodobieństwo Wyniku (%)")
    return fig


# --- 2. INTERFEJS ---

st.title("⚽ Advanced AI Predictor 2.0")

# --- PANEL BOCZNY ---
st.sidebar.header("Konfiguracja")

# Konfiguracja z Backtestu
league_config = {
    "Bundesliga": {"roi_1x2": 0.29, "roi_ou": -14.85, "recom": "🟡 MAŁY ZYSK (Ostrożnie)"},
    "Bundesliga 2": {"roi_1x2": -9.44, "roi_ou": -25.80, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Championship": {"roi_1x2": 14.31, "roi_ou": -3.81, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Eredivisie": {"roi_1x2": 26.59, "roi_ou": -18.47, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Greece Super League": {"roi_1x2": 38.11, "roi_ou": 20.05, "recom": "🔥 GRAJ WSZYSTKO"},
    "Jupiler League": {"roi_1x2": -29.22, "roi_ou": 9.95, "recom": "🔥 GRAJ GOLE (O/U)"},
    "La Liga": {"roi_1x2": 5.95, "roi_ou": 44.56, "recom": "🔥 GRAJ GOLE (O/U)"},
    "La Liga 2": {"roi_1x2": -14.14, "roi_ou": 9.80, "recom": "🔥 GRAJ GOLE (O/U)"},
    "League One": {"roi_1x2": 6.36, "roi_ou": -14.27, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "League Two": {"roi_1x2": -15.54, "roi_ou": 11.74, "recom": "🔥 GRAJ GOLE (O/U)"},
    "Liga Portugal": {"roi_1x2": -18.10, "roi_ou": -0.18, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Ligue 1": {"roi_1x2": -43.09, "roi_ou": -9.61, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Ligue 2": {"roi_1x2": -36.91, "roi_ou": -28.99, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Premier League": {"roi_1x2": -9.73, "roi_ou": -22.39, "recom": "⛔ UNIKAJ TEJ LIGI"},
    "Scottish Premiership": {"roi_1x2": 34.72, "roi_ou": -9.43, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Serie A": {"roi_1x2": 49.42, "roi_ou": -15.18, "recom": "🔥 GRAJ ZWYCIĘZCĘ (1X2)"},
    "Serie B": {"roi_1x2": -20.08, "roi_ou": 13.05, "recom": "🔥 GRAJ GOLE (O/U)"},
    "Super Lig": {"roi_1x2": -1.37, "roi_ou": 17.20, "recom": "🔥 GRAJ GOLE (O/U)"},
}

selected_league = st.sidebar.selectbox("Wybierz Ligę", sorted(list(league_config.keys())))

# Wyświetlanie rekomendacji
cfg = league_config.get(selected_league, {"roi_1x2": 0, "roi_ou": 0, "recom": "?"})
st.sidebar.info(f"Rekomendacja: {cfg['recom']}")
c1, c2 = st.sidebar.columns(2)
c1.metric("ROI 1X2", f"{cfg['roi_1x2']}%")
c2.metric("ROI O/U", f"{cfg['roi_ou']}%")

st.sidebar.divider()
bet_type = st.sidebar.radio("Tryb", ["Zwycięzca (1X2)", "Gole (Over/Under 2.5)"])

# ŁADOWANIE
with st.spinner(f'Analiza: {selected_league}...'):
    raw_data = load_and_prep_data(selected_league)
    if raw_data.empty: st.error("Brak danych."); st.stop()
    processed_data = add_rolling_features(raw_data)

    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, prediction_type=train_mode)

# TERMINARZ
st.sidebar.divider()
st.sidebar.header("Mecz")
fixtures = get_upcoming_fixtures(selected_league)
match_opts = ["Wybierz ręcznie..."]
match_map = {}
if not fixtures.empty:
    for idx, row in fixtures.iterrows():
        lbl = f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} vs {row['AwayTeam']}"
        match_opts.append(lbl)
        match_map[lbl] = row

sel_fix = st.sidebar.selectbox("Terminarz", match_opts)

# Domyślne wartości
def_h, def_a = 0, 1
d_o1, d_ox, d_o2 = 2.0, 3.2, 3.5
d_oo, d_ou = 1.9, 1.9

if sel_fix != "Wybierz ręcznie...":
    md = match_map[sel_fix]
    teams = sorted(raw_data['HomeTeam'].unique())
    try:
        def_h = teams.index(md['HomeTeam'])
        def_a = teams.index(md['AwayTeam'])
        if pd.notnull(md.get('B365H')): d_o1 = float(md['B365H'])
        if pd.notnull(md.get('B365D')): d_ox = float(md['B365D'])
        if pd.notnull(md.get('B365A')): d_o2 = float(md['B365A'])
        if pd.notnull(md.get('B365>2.5')): d_oo = float(md['B365>2.5'])
        if pd.notnull(md.get('B365<2.5')): d_ou = float(md['B365<2.5'])
    except:
        pass

teams = sorted(raw_data['HomeTeam'].unique())
home_team = st.sidebar.selectbox("Gospodarz", teams, index=def_h)
away_team = st.sidebar.selectbox("Gość", teams, index=def_a)

if bet_type == "Zwycięzca (1X2)":
    st.sidebar.subheader("Kursy 1X2")
    o1 = st.sidebar.number_input("1 (Home)", value=d_o1, step=0.05)
    ox = st.sidebar.number_input("X (Draw)", value=d_ox, step=0.05)
    o2 = st.sidebar.number_input("2 (Away)", value=d_o2, step=0.05)
else:
    st.sidebar.subheader("Kursy O/U")
    oo = st.sidebar.number_input("Over 2.5", value=d_oo, step=0.05)
    ou = st.sidebar.number_input("Under 2.5", value=d_ou, step=0.05)

st.sidebar.divider()
bankroll = st.sidebar.number_input("Budżet", value=1000)
kelly_frac = st.sidebar.select_slider("Kelly", [0.05, 0.1, 0.2], value=0.1)

# --- 3. GŁÓWNA ANALIZA ---
if st.sidebar.button("PRZEANALIZUJ MECZ", type="primary"):
    try:
        h_stats = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stats = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        input_data = pd.DataFrame([{
            'OddsDiff': (1 / (o1 if bet_type == "Zwycięzca (1X2)" else oo)) - (
                        1 / (o2 if bet_type == "Zwycięzca (1X2)" else ou)),
            'Home_Att': h_stats['Home_Att'], 'Away_Att': a_stats['Away_Att'],
            'Home_Def': h_stats['Home_Def'], 'Away_Def': a_stats['Away_Def'],
            'Home_Form': h_stats['Home_Form'], 'Away_Form': a_stats['Away_Form'],
            'Home_Corners_Avg': h_stats['Home_Corners_Avg'], 'Away_Corners_Avg': a_stats['Away_Corners_Avg']
        }])

        if bet_type == "Zwycięzca (1X2)":
            input_data['B365H'] = o1
            input_data['B365D'] = ox
            input_data['B365A'] = o2
            probs = model.predict_proba(input_data)[0]
            outcomes = [("GOSPODARZ (1)", probs[2], o1), ("REMIS (X)", probs[1], ox), ("GOŚĆ (2)", probs[0], o2)]
        else:
            input_data['B365_O25'] = oo
            input_data['B365_U25'] = ou
            p_over = model.predict_proba(input_data)[0][1]
            outcomes = [("OVER 2.5", p_over, oo), ("UNDER 2.5", 1 - p_over, ou)]

        st.subheader(f"{home_team} vs {away_team}")
        cols = st.columns(len(outcomes))

        for i, (lbl, prob, odd) in enumerate(outcomes):
            with cols[i]:
                fair = 1 / prob if prob > 0 else 0
                st.metric(lbl, f"{prob * 100:.1f}%", f"Fair: {fair:.2f}", delta_color="off")
                val = (prob * odd) - 1
                stake = calculate_kelly(prob, odd, bankroll, kelly_frac)

                if val > 0.05:
                    st.success(f"📈 VALUE: {val * 100:.1f}%")
                    if stake > 0: st.markdown(f"Stawka: **{stake:.2f} PLN**")
                elif val > 0:
                    st.warning(f"Małe Value: {val * 100:.1f}%")
                else:
                    st.error("Brak Value")
                st.caption(f"Bukmacher: {odd:.2f}")

        # HEATMAPA I xG
        st.divider()
        xg_h = (h_stats['Home_Att'] + a_stats['Away_Def']) / 2
        xg_a = (a_stats['Away_Att'] + h_stats['Home_Def']) / 2

        c1, c2 = st.columns([2, 1])
        with c1:
            st.write("Heatmapa Wyników:")
            st.pyplot(plot_score_heatmap(xg_h, xg_a))
        with c2:
            st.info(f"xG: {xg_h:.2f} - {xg_a:.2f}")
            best_p, best_s = 0, (0, 0)
            for h in range(5):
                for a in range(5):
                    p = poisson.pmf(h, xg_h) * poisson.pmf(a, xg_a)
                    if p > best_p: best_p, best_s = p, (h, a)
            st.success(f"Typ: {best_s[0]}-{best_s[1]}")

    except Exception as e:
        st.error(f"Błąd: {e}")