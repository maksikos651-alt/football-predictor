import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson
import plotly.graph_objects as go

st.set_page_config(page_title="AI Football Sniper Dynamic", layout="wide", page_icon="⚽")


# --- 1. FUNKCJE SILNIKA ---

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


@st.cache_data(ttl=86400)
def load_all_data():
    try:
        db_url = st.secrets["DB_URL"]
        engine = create_engine(db_url)
    except:
        st.error("Brak secrets.toml!");
        st.stop()

    query = "SELECT * FROM matches ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC',
        'home_shots': 'HS', 'away_shots': 'AS',
        'home_yellow': 'HY', 'away_yellow': 'AY',
        'home_fouls': 'HF', 'away_fouls': 'AF',
        'league': 'League'
    })
    return add_rolling_features(df)


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


# --- NOWA FUNKCJA DYNAMICZNEGO ROI ---
def calculate_real_roi(df, bet_type):
    """
    Oblicza UCZCIWE ROI, trenując model na przeszłości (80%)
    i testując na przyszłości (20%).
    """
    # Musimy mieć minimum danych, żeby to miało sens
    if len(df) < 100: return 0.0, 0

    # 1. Sortujemy chronologicznie
    df = df.sort_values("Date")

    # 2. Dzielimy na Trening (80%) i Test (20%)
    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:].copy()

    # 3. Trenujemy OSOBNY, tymczasowy model tylko na starej części
    # (Dzięki temu model nie zna wyników meczów, które obstawia)
    features = ['OddsDiff', 'Home_Att', 'Away_Att', 'Home_Def', 'Away_Def', 'Home_Form', 'Away_Form',
                'Home_Corners_Avg', 'Away_Corners_Avg', 'Home_Shots_Avg', 'Away_Shots_Avg', 'Home_Cards_Avg',
                'Away_Cards_Avg', 'Home_Fouls_Avg', 'Away_Fouls_Avg']

    temp_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.02, max_depth=3, n_jobs=1)

    if bet_type == "1X2":
        # Trening
        target = train_df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        cols = features + ['B365H', 'B365A', 'B365D']
        temp_model.fit(train_df[features], target)
    else:
        # Trening O/U
        target = np.where((train_df['FTHG'] + train_df['FTAG']) > 2.5, 1, 0)
        temp_model.fit(train_df[features], target)

    # 4. Symulacja na części TESTOWEJ (ostatnie 20%)
    bankroll = 1000.0
    stakes_sum = 0
    profit = 0

    for idx, row in test_df.iterrows():
        # Rekonstrukcja danych (tak jak w historii)
        input_dict = {col: row[col] for col in features if col in row}
        try:
            if bet_type == "1X2":
                if pd.isna(row['B365H']): continue
                input_dict['OddsDiff'] = (1 / row['B365H']) - (1 / row['B365A'])
            else:
                if pd.isna(row['B365_O25']): continue
                input_dict['OddsDiff'] = (1 / row['B365_O25']) - (1 / row['B365_U25'])
        except:
            continue

        clean_input = pd.DataFrame([input_dict])[features]

        # Predykcja
        if bet_type == "1X2":
            probs = temp_model.predict_proba(clean_input)[0]
            outcomes = [(2, probs[2], row['B365H']), (0, probs[0], row['B365A'])]  # Home, Away
        else:
            p_over = temp_model.predict_proba(clean_input)[0][1]
            outcomes = [(1, p_over, row['B365_O25']), (0, 1 - p_over, row['B365_U25'])]  # Over, Under

        # Szukamy Value
        best_val = 0
        final_stake = 0
        win_amount = 0

        for target, prob, odd in outcomes:
            if odd > 5.00: continue  # Ignorujemy kursy > 5.00 (zbyt ryzykowne)
            val = (prob * odd) - 1
            if val > 0.05:  # Próg 5%
                stake = calculate_kelly(prob, odd, 1000, 0.1)
                if stake > 0:
                    final_stake = stake
                    # Sprawdzamy czy weszło
                    if bet_type == "1X2":
                        real = 2 if row['FTR'] == 'H' else (0 if row['FTR'] == 'A' else 1)
                    else:
                        real = 1 if (row['FTHG'] + row['FTAG']) > 2.5 else 0

                    if real == target:
                        win_amount = (stake * odd) - stake
                    else:
                        win_amount = -stake
                    break  # Gramy tylko 1 typ na mecz

        if final_stake > 0:
            stakes_sum += final_stake
            profit += win_amount

    roi = (profit / stakes_sum * 100) if stakes_sum > 0 else 0.0
    return roi, len(test_df)


def get_h2h_history(df, team1, team2):
    mask = ((df['HomeTeam'] == team1) & (df['AwayTeam'] == team2)) | \
           ((df['HomeTeam'] == team2) & (df['AwayTeam'] == team1))
    h2h = df[mask].sort_values('Date', ascending=False).head(5)
    return h2h


def plot_radar_chart(h_stats, a_stats, home_name, away_name):
    categories = ['Atak', 'Obrona', 'Rożne', 'Strzały', 'Kartki', 'Faule']
    h_values = [h_stats['Home_Att'], h_stats['Home_Def'], h_stats['Home_Corners_Avg'], h_stats['Home_Shots_Avg'] / 3,
                h_stats['Home_Cards_Avg'] * 2, h_stats['Home_Fouls_Avg'] / 3]
    a_values = [a_stats['Away_Att'], a_stats['Away_Def'], a_stats['Away_Corners_Avg'], a_stats['Away_Shots_Avg'] / 3,
                a_stats['Away_Cards_Avg'] * 2, a_stats['Away_Fouls_Avg'] / 3]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=h_values, theta=categories, fill='toself', name=home_name, line_color='blue'))
    fig.add_trace(go.Scatterpolar(r=a_values, theta=categories, fill='toself', name=away_name, line_color='red'))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, max(max(h_values), max(a_values)) + 1])),
                      height=350, margin=dict(l=40, r=40, t=20, b=20))
    return fig


def plot_score_heatmap(home_exp, away_exp):
    probs = np.zeros((5, 5))
    for h in range(5):
        for a in range(5):
            probs[a, h] = poisson.pmf(h, home_exp) * poisson.pmf(a, away_exp) * 100
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(probs, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel("Gospodarz");
    ax.set_ylabel("Gość");
    ax.set_title("Dokładny Wynik")
    return fig


# --- 3. UI ---

st.title("⚽ AI Football Sniper Dynamic")

# Sidebar Configuration
st.sidebar.header("Konfiguracja")
available_leagues = [
    "Premier League", "Championship", "League One", "League Two",
    "La Liga", "La Liga 2", "Bundesliga", "Bundesliga 2",
    "Serie A", "Serie B", "Ligue 1", "Ligue 2",
    "Eredivisie", "Liga Portugal", "Jupiler League",
    "Super Lig", "Greece Super League", "Scottish Premiership"
]
selected_league = st.sidebar.selectbox("Liga", sorted(available_leagues))
bet_type = st.sidebar.radio("Strategia", ["Zwycięzca (1X2)", "Gole (Over/Under 2.5)"])
bankroll = st.sidebar.number_input("Bankroll", 1000)
kelly_frac = st.sidebar.slider("Kelly %", 0.05, 0.2, 0.1)

# DATA LOADING & TRAINING
with st.spinner("Przetwarzanie danych..."):
    all_data = load_all_data()
    if all_data.empty: st.error("Błąd bazy"); st.stop()
    processed_data = all_data[all_data['League'] == selected_league].copy()
    raw_data = processed_data
    if processed_data.empty: st.warning("Brak danych dla ligi"); st.stop()

    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, train_mode)

# DYNAMIC ROI CALCULATION
# --- ROI JEST TERAZ LICZONE NA OSOBNYM ZBIORZE TESTOWYM (BEZ OSZUKIWANIA) ---
roi_live, games_count = calculate_real_roi(processed_data, "1X2" if bet_type == "Zwycięzca (1X2)" else "OU")

st.sidebar.divider()
st.sidebar.subheader("📊 Uczciwe ROI (Backtest)")
st.sidebar.caption(f"Wynik z ostatnich {games_count} meczów (których model nie widział podczas treningu):")

if roi_live > 0:
    st.sidebar.success(f"📈 +{roi_live:.2f}%")
else:
    st.sidebar.error(f"📉 {roi_live:.2f}%")

# TABS
tab1, tab2, tab3 = st.tabs(["🔥 Skaner", "🔍 Analiza", "📜 Historia"])

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
                        {"Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}", "Typ": nm, "Kurs": odd,
                         "Szansa": f"{p * 100:.1f}%", "Value": f"{val * 100:.1f}%"})
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
                    {"Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}", "Typ": "OVER 2.5", "Kurs": oo,
                     "Szansa": f"{po * 100:.1f}%", "Value": f"{vo * 100:.1f}%"})
                if vu > 0.05: value_bets.append(
                    {"Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}", "Typ": "UNDER 2.5", "Kurs": ou,
                     "Szansa": f"{pu * 100:.1f}%", "Value": f"{vu * 100:.1f}%"})

        progress.empty()
        if value_bets:
            st.dataframe(pd.DataFrame(value_bets), use_container_width=True)
        else:
            st.warning("Brak Value > 5%")

with tab2:
    st.header("Analiza")
    match_opts = ["Wybierz ręcznie..."]
    match_map = {}
    if not fixtures.empty:
        for idx, row in fixtures.iterrows():
            lbl = f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} vs {row['AwayTeam']}"
            match_opts.append(lbl)
            match_map[lbl] = row
    sel_fix = st.selectbox("Mecz", match_opts, key="t2_sel")
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

    st.divider()
    h2h_data = get_h2h_history(raw_data, home_team, away_team)
    if not h2h_data.empty:
        st.caption("Ostatnie mecze H2H:")
        for idx, row in h2h_data.iterrows(): st.text(
            f"{row['Date'].strftime('%d.%m')} | {row['HomeTeam']} {int(row['FTHG'])} - {int(row['FTAG'])} {row['AwayTeam']}")

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

    if st.button("ANALIZA", type="primary"):
        h_stat = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stat = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        c_rad, c_line = st.columns(2)
        c_rad.plotly_chart(plot_radar_chart(h_stat, a_stat, home_team, away_team), use_container_width=True)
        h_vals = processed_data[processed_data['HomeTeam'] == home_team].tail(10)['Home_Form'].values
        a_vals = processed_data[processed_data['AwayTeam'] == away_team].tail(10)['Away_Form'].values
        min_len = min(len(h_vals), len(a_vals))
        if min_len > 1:
            chart = pd.DataFrame({f"{home_team}": h_vals[-min_len:], f"{away_team}": a_vals[-min_len:]})
            chart.index = range(1, min_len + 1)
            c_line.line_chart(chart)

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
            input_data['OddsDiff'] = (1 / o1) - (1 / o2);
            input_data['B365H'] = o1;
            input_data['B365D'] = ox;
            input_data['B365A'] = o2
            input_data = input_data[features]
            probs = model.predict_proba(input_data)[0]
            outcomes = [("1", probs[2], o1), ("X", probs[1], ox), ("2", probs[0], o2)]
        else:
            input_data['OddsDiff'] = (1 / oo) - (1 / ou);
            input_data['B365_O25'] = oo;
            input_data['B365_U25'] = ou
            input_data = input_data[features]
            po = model.predict_proba(input_data)[0][1]
            outcomes = [("Over", po, oo), ("Under", 1 - po, ou)]

        c_res = st.columns(len(outcomes))
        for i, (lbl, p, o) in enumerate(outcomes):
            with c_res[i]:
                val = (p * o) - 1
                st.metric(lbl, f"{p * 100:.1f}%", f"Val: {val * 100:.1f}%", delta_color="normal" if val > 0 else "off")
                if val > 0.05: st.success(f"Stawka: {calculate_kelly(p, o, bankroll, kelly_frac):.2f}")

        xg_h = (h_stat['Home_Att'] + a_stat['Away_Def']) / 2
        xg_a = (a_stat['Away_Att'] + h_stat['Home_Def']) / 2
        st.pyplot(plot_score_heatmap(xg_h, xg_a))
        st.info(f"xG: {xg_h:.2f} - {xg_a:.2f}")

# --- TAB 3: HISTORIA (UCZCIWA WERYFIKACJA) ---
with tab3:
    st.header("Weryfikacja (Backtest na ostatnich 50 meczach)")
    st.info("Symulacja: Model został wytrenowany na danych SPRZED tych 50 meczów. Nie znał ich wyników.")

    # 1. Przygotowanie danych (sortowanie chronologiczne)
    df_sorted = processed_data.sort_values("Date", ascending=True)

    # Zabezpieczenie - musimy mieć na czym trenować
    if len(df_sorted) < 100:
        st.warning("Za mało danych w bazie, by przeprowadzić uczciwy test (potrzeba min. 100 meczów historycznych).")
    else:
        # 2. Podział: Ostatnie 50 to TEST, reszta to TRENING
        test_size = 50
        train_df = df_sorted.iloc[:-test_size]
        test_df = df_sorted.iloc[-test_size:].copy()  # To są te mecze, które pokażemy w tabeli

        # 3. Trening modelu "Historycznego" (który nie zna przyszłości)
        # Używamy lekkich parametrów (n_estimators=150), żeby było szybko
        history_model = xgb.XGBClassifier(n_estimators=150, learning_rate=0.02, max_depth=3, n_jobs=1)

        if bet_type == "Zwycięzca (1X2)":
            target = train_df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
            history_model.fit(train_df[features], target)
        else:
            target = np.where((train_df['FTHG'] + train_df['FTAG']) > 2.5, 1, 0)
            history_model.fit(train_df[features], target)

        # 4. Predykcja na zbiorze testowym (odwracamy kolejność, żeby najnowsze były na górze)
        rows = []
        # Sortujemy test_df od najnowszych do wyświetlania
        test_view = test_df.sort_values("Date", ascending=False)

        for idx, row in test_view.iterrows():
            input_dict = {col: row[col] for col in features if col in row}
            try:
                if bet_type == "Zwycięzca (1X2)":
                    if pd.isna(row['B365H']) or pd.isna(row['B365A']): continue
                    input_dict['OddsDiff'] = (1 / row['B365H']) - (1 / row['B365A'])
                else:
                    if pd.isna(row['B365_O25']) or pd.isna(row['B365_U25']): continue
                    input_dict['OddsDiff'] = (1 / row['B365_O25']) - (1 / row['B365_U25'])
            except:
                continue

            clean = pd.DataFrame([input_dict])[features]

            pick, odd, val_perc, res = "PAS", 0.0, 0.0, "⚪"

            if bet_type == "Zwycięzca (1X2)":
                probs = history_model.predict_proba(clean)[0]
                vh = (probs[2] * row['B365H']) - 1
                va = (probs[0] * row['B365A']) - 1

                if vh > 0.05:
                    pick = "HOME";
                    odd = row['B365H'];
                    val_perc = vh;
                    res = "🟢 WIN" if row['FTR'] == 'H' else "🔴 LOSS"
                elif va > 0.05:
                    pick = "AWAY";
                    odd = row['B365A'];
                    val_perc = va;
                    res = "🟢 WIN" if row['FTR'] == 'A' else "🔴 LOSS"
            else:
                po = history_model.predict_proba(clean)[0][1]
                vo = (po * row['B365_O25']) - 1
                vu = ((1 - po) * row['B365_U25']) - 1
                gls = row['FTHG'] + row['FTAG']

                if vo > 0.05:
                    pick = "OVER";
                    odd = row['B365_O25'];
                    val_perc = vo;
                    res = "🟢 WIN" if gls > 2.5 else "🔴 LOSS"
                elif vu > 0.05:
                    pick = "UNDER";
                    odd = row['B365_U25'];
                    val_perc = vu;
                    res = "🟢 WIN" if gls < 2.5 else "🔴 LOSS"

            # Dodajemy tylko jeśli była decyzja (żeby nie śmiecić tabeli PAS-ami)
            if pick != "PAS":
                rows.append({
                    "Data": row['Date'].strftime('%d.%m'),
                    "Mecz": f"{row['HomeTeam']} vs {row['AwayTeam']}",
                    "Wynik": f"{int(row['FTHG'])}-{int(row['FTAG'])}" if pd.notnull(row['FTHG']) else "?-?",
                    "AI": pick,
                    "Kurs": f"{odd:.2f}",
                    "Value": f"{val_perc * 100:.1f}%",
                    "Status": res
                })

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        else:
            st.warning("Model historyczny nie znalazłby żadnej okazji > 5% Value w ostatnich 50 meczach.")