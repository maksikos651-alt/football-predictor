import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import plotly.graph_objects as go  # <--- NOWOŚĆ

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="Typy Grezegorza", layout="wide", page_icon="⚽")


# --- 1. FUNKCJE SILNIKA (Backend) ---

@st.cache_data(ttl=3600)
def get_upcoming_fixtures(league_name):
    url = "https://www.football-data.co.uk/fixtures.csv"
    try:
        df = pd.read_csv(url, encoding='utf-8-sig', on_bad_lines='skip')
        df.columns = df.columns.str.strip()
        if 'Div' not in df.columns:
            for col in df.columns:
                if 'Div' in col or 'Division' in col: df = df.rename(columns={col: 'Div'}); break
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


@st.cache_data(ttl=86400)  # <--- ZMIANA: Pamięć na 24 godziny!
def load_all_data():
    """Pobiera całą bazę danych naraz i oblicza statystyki dla wszystkich lig."""
    try:
        db_url = st.secrets["DB_URL"]
        engine = create_engine(db_url)
    except:
        # Fallback lokalny
        DB_USER = 'postgres'
        DB_PASS = 'EAtocepy12!'
        DB_HOST = 'localhost'
        engine = create_engine(f'postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:5432/postgres')

    # Pobieramy WSZYSTKO (bez filtrowania WHERE league=...)
    query = "SELECT * FROM matches ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    # Zmiana nazw kolumn
    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC',
        'home_shots': 'HS', 'away_shots': 'AS',
        'home_yellow': 'HY', 'away_yellow': 'AY',
        'home_fouls': 'HF', 'away_fouls': 'AF',
        'league': 'League'  # Ważne: zachowujemy kolumnę League do filtrowania później
    })

    # --- PRZELICZANIE STATYSTYK (ROLLING) DLA CAŁEJ BAZY RAZEM ---
    # Dzięki temu robimy to raz, a nie przy każdym kliknięciu
    processed_df = add_rolling_features(df)

    return processed_df


def get_h2h_history(df, team1, team2):
    mask = ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | \
           ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    h2h = df[mask].sort_values('Date', ascending=False).head(5)
    return h2h


def plot_radar_chart(h_stats, a_stats, home_name, away_name):
    categories = ['Atak (Gole)', 'Obrona (Stracone)', 'Rożne', 'Strzały (skala /3)', 'Kartki (skala x2)',
                  'Faule (skala /3)']

    # Skalowanie dla czytelności wykresu
    h_values = [
        h_stats['Home_Att'], h_stats['Home_Def'], h_stats['Home_Corners_Avg'],
        h_stats['Home_Shots_Avg'] / 3, h_stats['Home_Cards_Avg'] * 2, h_stats['Home_Fouls_Avg'] / 3
    ]
    a_values = [
        a_stats['Away_Att'], a_stats['Away_Def'], a_stats['Away_Corners_Avg'],
        a_stats['Away_Shots_Avg'] / 3, a_stats['Away_Cards_Avg'] * 2, a_stats['Away_Fouls_Avg'] / 3
    ]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_values, theta=categories, fill='toself', name=home_name, line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=a_values, theta=categories, fill='toself', name=away_name, line_color='red'))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(h_values), max(a_values)) + 1])),
        margin=dict(l=40, r=40, t=40, b=40),
        height=400
    )
    return fig


def add_rolling_features(df, window=5):
    data = df.copy()
    for col in ['HS', 'AS', 'HY', 'AY', 'HF', 'AF']:
        if col not in data.columns: data[col] = 0

    home_df = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC', 'HS', 'AS', 'HY', 'HF']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GS', 'FTAG': 'GC', 'HC': 'CorW', 'AC': 'CorL', 'HS': 'ShotsW',
                 'AS': 'ShotsL', 'HY': 'Cards', 'HF': 'Fouls'})
    home_df['Pts'] = np.where(home_df['FTR'] == 'H', 3, np.where(home_df['FTR'] == 'D', 1, 0))
    home_df['IsHome'] = 1

    away_df = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AC', 'HC', 'AS', 'HS', 'AY', 'AF']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GS', 'FTHG': 'GC', 'AC': 'CorW', 'HC': 'CorL', 'AS': 'ShotsW',
                 'HS': 'ShotsL', 'AY': 'Cards', 'AF': 'Fouls'})
    away_df['Pts'] = np.where(away_df['FTR'] == 'A', 3, np.where(away_df['FTR'] == 'D', 1, 0))
    away_df['IsHome'] = 0

    team_stats = pd.concat([home_df, away_df]).sort_values('Date')
    features = team_stats.groupby('Team')[['GS', 'GC', 'Pts', 'CorW', 'ShotsW', 'Cards', 'Fouls']].transform(
        lambda x: x.rolling(window, min_periods=3).mean().shift(1)
    )
    team_stats[['Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor', 'Form_Shots', 'Form_Cards', 'Form_Fouls']] = features

    cols_to_keep = ['Date', 'Team', 'Form_Att', 'Form_Def', 'Form_Pts', 'Form_Cor', 'Form_Shots', 'Form_Cards',
                    'Form_Fouls']
    h_stats = team_stats[team_stats['IsHome'] == 1][cols_to_keep].rename(
        columns={'Team': 'HomeTeam', 'Form_Att': 'Home_Att', 'Form_Def': 'Home_Def', 'Form_Pts': 'Home_Form',
                 'Form_Cor': 'Home_Corners_Avg', 'Form_Shots': 'Home_Shots_Avg', 'Form_Cards': 'Home_Cards_Avg',
                 'Form_Fouls': 'Home_Fouls_Avg'})
    a_stats = team_stats[team_stats['IsHome'] == 0][cols_to_keep].rename(
        columns={'Team': 'AwayTeam', 'Form_Att': 'Away_Att', 'Form_Def': 'Away_Def', 'Form_Pts': 'Away_Form',
                 'Form_Cor': 'Away_Corners_Avg', 'Form_Shots': 'Away_Shots_Avg', 'Form_Cards': 'Away_Cards_Avg',
                 'Form_Fouls': 'Away_Fouls_Avg'})

    data = pd.merge(data, h_stats, on=['Date', 'HomeTeam'], how='left')
    data = pd.merge(data, a_stats, on=['Date', 'AwayTeam'], how='left')
    data['OddsDiff'] = (1 / data['B365H']) - (1 / data['B365A'])
    return data.dropna()


@st.cache_resource
def train_model(df, prediction_type="1X2"):
    predictors = ['OddsDiff', 'Home_Att', 'Away_Att', 'Home_Def', 'Away_Def', 'Home_Form', 'Away_Form',
                  'Home_Corners_Avg', 'Away_Corners_Avg', 'Home_Shots_Avg', 'Away_Shots_Avg', 'Home_Cards_Avg',
                  'Away_Cards_Avg', 'Home_Fouls_Avg', 'Away_Fouls_Avg']
    best_params = {'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4, 'subsample': 0.7,
                   'colsample_bytree': 0.8, 'n_jobs': 1}

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
    "Serie B": {"roi_1x2": 35.46, "roi_ou": 6.29, "recom": "🔥 GRAJ WSZYSTKO"},
    "Bundesliga": {"roi_1x2": 23.58, "roi_ou": 5.63, "recom": "🔥 GRAJ WSZYSTKO (Nowy Król!)"},
    "Greece Super League": {"roi_1x2": 15.68, "roi_ou": 34.88, "recom": "🔥 GRAJ WSZYSTKO"},
    "Championship": {"roi_1x2": 12.55, "roi_ou": 14.51, "recom": "🔥 GRAJ WSZYSTKO"},
    "Serie A": {"roi_1x2": 13.97, "roi_ou": -4.63, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "La Liga 2": {"roi_1x2": 16.89, "roi_ou": 4.25, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "Scottish Premiership": {"roi_1x2": 9.43, "roi_ou": -44.03, "recom": "✅ GRAJ ZWYCIĘZCĘ (1X2)"},
    "Jupiler League": {"roi_1x2": -31.89, "roi_ou": 23.39, "recom": "✅ GRAJ GOLE (O/U)"},
    "Super Lig": {"roi_1x2": -2.77, "roi_ou": 8.82, "recom": "✅ GRAJ GOLE (O/U)"},
    "Premier League": {"roi_1x2": -16.94, "roi_ou": -29.05, "recom": "⛔ UNIKAJ (Strata pieniędzy)"},
    "La Liga": {"roi_1x2": -17.99, "roi_ou": -13.01, "recom": "⛔ UNIKAJ (Za trudna)"},
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

cfg = league_config.get(selected_league, {"roi_1x2":0,"roi_ou":0,"recom":"?"})
st.sidebar.info(f"{cfg['recom']}")

# --- Wklej to, żeby przywrócić ROI ---
col_roi1, col_roi2 = st.sidebar.columns(2)
col_roi1.metric("ROI 1X2", f"{cfg['roi_1x2']}%", delta_color="normal" if cfg['roi_1x2'] > 0 else "off")
col_roi2.metric("ROI O/U", f"{cfg['roi_ou']}%", delta_color="normal" if cfg['roi_ou'] > 0 else "off")
# -------------------------------------
# --- OPTYMALIZACJA ŁADOWANIA ---
with st.spinner("Ładowanie i przetwarzanie danych (to zdarza się raz na 24h)..."):
    # 1. Pobieramy WSZYSTKO (z Cache)
    all_data = load_all_data()

    if all_data.empty: st.error("Błąd: Baza danych jest pusta."); st.stop()

    # 2. Filtrujemy wybraną ligę z pamięci RAM (Błyskawicznie)
    processed_data = all_data[all_data['League'] == selected_league].copy()

    if processed_data.empty: st.warning(f"Brak danych dla ligi: {selected_league}"); st.stop()

    # --- POPRAWKA: TWORZYMY KOPIĘ DLA KOMPATYBILNOŚCI ---
    # Dzięki temu reszta kodu (tabele, listy drużyn) widzi zmienną "raw_data" i nie wyrzuca błędu
    raw_data = processed_data
    # ----------------------------------------------------

    # 3. Trenujemy model tylko na wycinku danych (Szybko)
    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, train_mode)
# --- ZAKŁADKI ---
tab1, tab2, tab3 = st.tabs(["🔥 Skaner Value", "🔍 Analiza Meczu", "📜 Historia"])

# --- TAB 1: SKANER ---
with tab1:
    st.header(f"Skaner: {selected_league}")
    fixtures = get_upcoming_fixtures(selected_league)
    if fixtures.empty:
        st.info("Brak meczów.")
    else:
        value_bets = []
        progress = st.progress(0)
        total = len(fixtures)
        for i, (idx, row) in enumerate(fixtures.iterrows()):
            progress.progress(min((i + 1) / total, 1.0))
            h_stats = processed_data[processed_data['HomeTeam'] == row['HomeTeam']]
            a_stats = processed_data[processed_data['AwayTeam'] == row['AwayTeam']]
            if h_stats.empty or a_stats.empty: continue

            h_stat, a_stat = h_stats.iloc[-1], a_stats.iloc[-1]
            mi = pd.DataFrame([{
                'Home_Att': h_stat['Home_Att'], 'Away_Att': a_stat['Away_Att'],
                'Home_Def': h_stat['Home_Def'], 'Away_Def': a_stat['Away_Def'],
                'Home_Form': h_stat['Home_Form'], 'Away_Form': a_stat['Away_Form'],
                'Home_Corners_Avg': h_stat['Home_Corners_Avg'], 'Away_Corners_Avg': a_stat['Away_Corners_Avg'],
                'Home_Shots_Avg': h_stat['Home_Shots_Avg'], 'Away_Shots_Avg': a_stat['Away_Shots_Avg'],
                'Home_Cards_Avg': h_stat['Home_Cards_Avg'], 'Away_Cards_Avg': a_stat['Away_Cards_Avg'],
                'Home_Fouls_Avg': h_stat['Home_Fouls_Avg'], 'Away_Fouls_Avg': a_stat['Away_Fouls_Avg']
            }])

            if bet_type == "Zwycięzca (1X2)":
                o1, ox, o2 = row.get('B365H'), row.get('B365D'), row.get('B365A')
                if pd.isna(o1) or pd.isna(ox) or pd.isna(o2): continue
                mi['OddsDiff'] = (1 / o1) - (1 / o2);
                mi['B365H'] = o1;
                mi['B365D'] = ox;
                mi['B365A'] = o2
                mi = mi[features]
                probs = model.predict_proba(mi)[0]
                for o, p, odd, nm in [(2, probs[2], o1, row['HomeTeam']), (0, probs[0], o2, row['AwayTeam'])]:
                    val = (p * odd) - 1
                    if val > 0.05: value_bets.append(
                        {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                         "Typ": nm, "Kurs": odd, "Szansa": f"{p * 100:.1f}%", "Value": f"{val * 100:.1f}%",
                         "Info": "🔥 GRAJ"})
            else:
                oo, ou = row.get('B365>2.5'), row.get('B365<2.5')
                if pd.isna(oo) or pd.isna(ou): continue
                mi['OddsDiff'] = (1 / oo) - (1 / ou);
                mi['B365_O25'] = oo;
                mi['B365_U25'] = ou
                mi = mi[features]
                po = model.predict_proba(mi)[0][1]
                pu = 1.0 - po
                vo, vu = (po * oo) - 1, (pu * ou) - 1
                if vo > 0.05: value_bets.append(
                    {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                     "Typ": "OVER 2.5", "Kurs": oo, "Szansa": f"{po * 100:.1f}%", "Value": f"{vo * 100:.1f}%",
                     "Info": "🔥 GRAJ"})
                if vu > 0.05: value_bets.append(
                    {"Data": row['Date'].strftime('%d.%m'), "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                     "Typ": "UNDER 2.5", "Kurs": ou, "Szansa": f"{pu * 100:.1f}%", "Value": f"{vu * 100:.1f}%",
                     "Info": "🔥 GRAJ"})
        progress.empty()
        if value_bets:
            st.dataframe(pd.DataFrame(value_bets), use_container_width=True)
        else:
            st.warning("Brak Value > 5%")

# --- TAB 2: ANALIZA SZCZEGÓŁOWA (ULEPSZONA) ---
with tab2:
    st.header("Centrum Analizy Meczu")
    match_opts = ["Wybierz ręcznie..."]
    match_map = {}
    if not fixtures.empty:
        for idx, row in fixtures.iterrows():
            lbl = f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} vs {row['AwayTeam']}"
            match_opts.append(lbl)
            match_map[lbl] = row

    sel_fix = st.selectbox("Wybierz mecz z terminarza", match_opts, key="tab2_select")
    if "last_fix_t2" not in st.session_state: st.session_state.last_fix_t2 = None
    if sel_fix != st.session_state.last_fix_t2:
        st.session_state.last_fix_t2 = sel_fix
        if sel_fix != "Wybierz ręcznie..." and sel_fix in match_map:
            md = match_map[sel_fix]
            st.session_state['t2_h'] = md['HomeTeam']
            st.session_state['t2_a'] = md['AwayTeam']


            def safe_get(key, default=2.0):
                try:
                    val = float(md.get(key, 0.0)); return val if val > 1.0 else default
                except:
                    return default


            st.session_state['k_1'] = safe_get('B365H', 2.0);
            st.session_state['k_x'] = safe_get('B365D', 3.2);
            st.session_state['k_2'] = safe_get('B365A', 3.5)
            st.session_state['k_o'] = safe_get('B365>2.5', 1.9);
            st.session_state['k_u'] = safe_get('B365<2.5', 1.9)

    c1, c2 = st.columns(2)
    teams = sorted(raw_data['HomeTeam'].unique())
    try:
        if 't2_h' in st.session_state and st.session_state.t2_h not in teams: del st.session_state.t2_h
        if 't2_a' in st.session_state and st.session_state.t2_a not in teams: del st.session_state.t2_a
    except:
        pass
    home_team = c1.selectbox("Gospodarz", teams, key="t2_h")
    away_team = c2.selectbox("Gość", teams, key="t2_a")

    # H2H
    st.divider()
    st.subheader("⚔️ Historia H2H")
    h2h_data = get_h2h_history(raw_data, home_team, away_team)
    if h2h_data.empty:
        st.info("Brak bezpośrednich meczów.")
    else:
        for idx, row in h2h_data.iterrows():
            st.text(
                f"{row['Date'].strftime('%d.%m.%Y')} | {row['HomeTeam']} {int(row['FTHG'])} - {int(row['FTAG'])} {row['AwayTeam']}")

    st.divider()
    st.write("Kursy:")
    k1, k2, k3 = st.columns(3)
    if bet_type == "Zwycięzca (1X2)":
        o1 = k1.number_input("1", value=2.0, min_value=1.01, step=0.05, key="k_1")
        ox = k2.number_input("X", value=3.2, min_value=1.01, step=0.05, key="k_x")
        o2 = k3.number_input("2", value=3.5, min_value=1.01, step=0.05, key="k_2")
    else:
        oo = k1.number_input("Over", value=1.9, min_value=1.01, step=0.05, key="k_o")
        ou = k2.number_input("Under", value=1.9, min_value=1.01, step=0.05, key="k_u")
        o1, o2, ox = 0, 0, 0

    if st.button("URUCHOM AI", type="primary", key="btn_ai"):
        h_stat = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stat = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        # WYKRESY
        st.subheader("📊 Analiza Siły (Radar) i Trendu")
        c_rad, c_line = st.columns(2)
        with c_rad:
            st.plotly_chart(plot_radar_chart(h_stat, a_stat, home_team, away_team), use_container_width=True)
        with c_line:
            h_vals = processed_data[processed_data['HomeTeam'] == home_team].tail(10)['Home_Form'].values
            a_vals = processed_data[processed_data['AwayTeam'] == away_team].tail(10)['Away_Form'].values
            min_len = min(len(h_vals), len(a_vals))
            if min_len > 1:
                chart_df = pd.DataFrame({f"{home_team}": h_vals[-min_len:], f"{away_team}": a_vals[-min_len:]})
                chart_df.index = range(1, min_len + 1)
                st.line_chart(chart_df)
                st.caption("Oś X: Mecze (10 = ostatni)")

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
            input_data = input_data[features]
            probs = model.predict_proba(input_data)[0]
            outcomes = [("1", probs[2], o1), ("X", probs[1], ox), ("2", probs[0], o2)]
        else:
            input_data['OddsDiff'] = (1 / oo) - (1 / ou)
            input_data['B365_O25'] = oo;
            input_data['B365_U25'] = ou
            input_data = input_data[features]
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
    st.header("Weryfikacja")
    history = processed_data.sort_values("Date", ascending=False).head(20).copy()
    rows = []
    for idx, row in history.iterrows():
        input_dict = {col: row[col] for col in features if col in row}
        if 'OddsDiff' not in input_dict: input_dict['OddsDiff'] = (1 / row['B365H']) - (1 / row['B365A'])
        clean = pd.DataFrame([input_dict])[features]

        pick, odd, res = "-", 0.0, "⚪"
        if bet_type == "Zwycięzca (1X2)":
            probs = model.predict_proba(clean)[0]
            if (probs[2] * row['B365H']) - 1 > 0.1:
                pick = "HOME"; odd = row['B365H']; res = "🟢" if row['FTR'] == 'H' else "🔴"
            elif (probs[0] * row['B365A']) - 1 > 0.1:
                pick = "AWAY"; odd = row['B365A']; res = "🟢" if row['FTR'] == 'A' else "🔴"
        else:
            po = model.predict_proba(clean)[0][1]
            if (po * row['B365_O25']) - 1 > 0.1:
                pick = "OVER"; odd = row['B365_O25']; res = "🟢" if (row['FTHG'] + row['FTAG']) > 2.5 else "🔴"
            elif ((1 - po) * row['B365_U25']) - 1 > 0.1:
                pick = "UNDER"; odd = row['B365_U25']; res = "🟢" if (row['FTHG'] + row['FTAG']) < 2.5 else "🔴"

        rows.append(
            {"Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}", "Wynik": f"{int(row['FTHG'])}-{int(row['FTAG'])}",
             "AI": pick, "Kurs": f"{odd:.2f}" if odd > 0 else "-", "Status": res})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)