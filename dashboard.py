import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sqlalchemy import create_engine
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import poisson

# --- KONFIGURACJA STRONY ---
st.set_page_config(page_title="AI Football Predictor", layout="wide")


# --- 1. FUNKCJE SILNIKA ---

@st.cache_data(ttl=3600)
def get_upcoming_fixtures(league_name):
    url = "https://www.football-data.co.uk/fixtures.csv"

    try:
        # 1. Wczytujemy (ignorując błędy linii)
        df = pd.read_csv(url, encoding='latin1', on_bad_lines='skip')

        # Usuwamy spacje z nazw kolumn (częsty błąd w plikach CSV)
        df.columns = df.columns.str.strip()

        # Mapa kodów lig
        league_map = {
            "Premier League": "E0", "Championship": "E1",
            "La Liga": "SP1", "Bundesliga": "D1",
            "Serie A": "I1", "Ligue 1": "F1"
        }
        div_code = league_map.get(league_name)

        # 2. Sprawdzamy czy mamy kolumnę Div (jak nie, to plik jest zły)
        if 'Div' not in df.columns:
            return pd.DataFrame()

        # 3. Filtrujemy ligę
        league_fixtures = df[df['Div'] == div_code].copy()

        # 4. Formatujemy datę
        league_fixtures['Date'] = pd.to_datetime(league_fixtures['Date'], dayfirst=True, errors='coerce')

        # 5. FILTR CZASU: Usuwamy mecze, które już były!
        # Bierzemy dzisiejszą datę (bez godziny)
        today = pd.Timestamp.now().normalize()
        # Zostawiamy tylko mecze od dzisiaj w górę
        league_fixtures = league_fixtures[league_fixtures['Date'] >= today]

        league_fixtures = league_fixtures.sort_values(['Date', 'Time'])

        return league_fixtures

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_and_prep_data(league_name):
    # POBIERAMY URL BAZY Z SEKRETÓW STREAMLIT CLOUD (Bezpiecznie!)
    db_url = st.secrets["DB_URL"]

    engine = create_engine(db_url)



    query = f"SELECT * FROM matches WHERE league = '{league_name}' ORDER BY date ASC"
    df = pd.read_sql(query, engine)

    df = df.rename(columns={
        'date': 'Date', 'home_team': 'HomeTeam', 'away_team': 'AwayTeam',
        'home_goals': 'FTHG', 'away_goals': 'FTAG', 'result': 'FTR',
        'home_shots_target': 'HST', 'away_shots_target': 'AST',
        'odds_home': 'B365H', 'odds_draw': 'B365D', 'odds_away': 'B365A',
        'odds_over25': 'B365_O25', 'odds_under25': 'B365_U25',
        'home_corners': 'HC', 'away_corners': 'AC',
        'home_red': 'HR', 'away_red': 'AR'
    })
    return df


def add_rolling_features(df, window=5):
    data = df.copy()

    # 1. Home
    home_df = data[['Date', 'HomeTeam', 'FTHG', 'FTAG', 'FTR', 'HC', 'AC']].rename(
        columns={'HomeTeam': 'Team', 'FTHG': 'GoalsScored', 'FTAG': 'GoalsConceded',
                 'HC': 'CornersWon', 'AC': 'CornersLost'}
    )
    home_df['Points'] = np.where(home_df['FTR'] == 'H', 3, np.where(home_df['FTR'] == 'D', 1, 0))
    home_df['IsHome'] = 1

    # 2. Away
    away_df = data[['Date', 'AwayTeam', 'FTAG', 'FTHG', 'FTR', 'AC', 'HC']].rename(
        columns={'AwayTeam': 'Team', 'FTAG': 'GoalsScored', 'FTHG': 'GoalsConceded',
                 'AC': 'CornersWon', 'HC': 'CornersLost'}
    )
    away_df['Points'] = np.where(away_df['FTR'] == 'A', 3, np.where(away_df['FTR'] == 'D', 1, 0))
    away_df['IsHome'] = 0

    # 3. Rolling
    team_stats = pd.concat([home_df, away_df]).sort_values('Date')
    features = team_stats.groupby('Team')[['GoalsScored', 'GoalsConceded', 'Points', 'CornersWon']].transform(
        lambda x: x.rolling(window, min_periods=3).mean().shift(1)
    )

    team_stats['Form_Goals'] = features['GoalsScored']
    team_stats['Form_Defense'] = features['GoalsConceded']
    team_stats['Form_Points'] = features['Points']
    team_stats['Form_Corners'] = features['CornersWon']

    # 4. Merge back
    stats_home = team_stats[team_stats['IsHome'] == 1][
        ['Date', 'Team', 'Form_Goals', 'Form_Defense', 'Form_Points', 'Form_Corners']]
    stats_home = stats_home.rename(
        columns={'Team': 'HomeTeam', 'Form_Goals': 'Home_Att', 'Form_Defense': 'Home_Def', 'Form_Points': 'Home_Form',
                 'Form_Corners': 'Home_Corners_Avg'})

    stats_away = team_stats[team_stats['IsHome'] == 0][
        ['Date', 'Team', 'Form_Goals', 'Form_Defense', 'Form_Points', 'Form_Corners']]
    stats_away = stats_away.rename(
        columns={'Team': 'AwayTeam', 'Form_Goals': 'Away_Att', 'Form_Defense': 'Away_Def', 'Form_Points': 'Away_Form',
                 'Form_Corners': 'Away_Corners_Avg'})

    data = pd.merge(data, stats_home, on=['Date', 'HomeTeam'], how='left')
    data = pd.merge(data, stats_away, on=['Date', 'AwayTeam'], how='left')

    data['OddsDiff'] = (1 / data['B365H']) - (1 / data['B365A'])

    return data.dropna()


@st.cache_resource
def train_model(df, prediction_type="1X2"):
    predictors = [
        'OddsDiff',
        'Home_Att', 'Away_Att',
        'Home_Def', 'Away_Def',
        'Home_Form', 'Away_Form',
        'Home_Corners_Avg', 'Away_Corners_Avg'
    ]

    # Parametry z Twojej optymalizacji
    best_params = {
        'n_estimators': 300, 'learning_rate': 0.01, 'max_depth': 4,
        'subsample': 0.7, 'colsample_bytree': 0.8, 'random_state': 42, 'n_jobs': -1
    }

    if prediction_type == "1X2":
        predictors += ['B365H', 'B365A', 'B365D']
        target_map = {'A': 0, 'D': 1, 'H': 2}
        df['Target_Multi'] = df['FTR'].map(target_map)

        model = xgb.XGBClassifier(objective='multi:softprob', num_class=3, **best_params)
        model.fit(df[predictors], df['Target_Multi'])

    elif prediction_type == "OU25":
        predictors += ['B365_O25', 'B365_U25']
        total_goals = df['FTHG'] + df['FTAG']
        target = np.where(total_goals > 2.5, 1, 0)

        model = xgb.XGBClassifier(objective='binary:logistic', **best_params)
        model.fit(df[predictors], target)

    return model, predictors


def calculate_kelly(prob, odds, bankroll, fraction):
    if prob * odds <= 1: return 0.0
    b = odds - 1
    p = prob
    q = 1 - p
    full = (b * p - q) / b
    return bankroll * max(full, 0) * fraction


def plot_score_heatmap(home_goal_exp, away_goal_exp):
    """Rysuje mapę prawdopodobieństwa dokładnych wyników"""
    max_goals = 5
    probs = np.zeros((max_goals, max_goals))

    for i in range(max_goals):  # Goście
        for j in range(max_goals):  # Gospodarze
            prob = poisson.pmf(j, home_goal_exp) * poisson.pmf(i, away_goal_exp)
            probs[i, j] = prob * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(probs, annot=True, fmt=".1f", cmap="YlGnBu", cbar=False, ax=ax)
    ax.set_xlabel(f"Gole Gospodarza")
    ax.set_ylabel(f"Gole Gościa")
    ax.set_title("Prawdopodobieństwo Dokładnego Wyniku (%)")
    return fig
# --- 2. INTERFEJS ---

st.title("⚽ Advanced AI Predictor (Auto-Fixtures)")

# SIDEBAR - KONFIGURACJA
st.sidebar.header("Konfiguracja")
selected_league = st.sidebar.selectbox("Liga", ["Premier League", "Championship", "La Liga", "Bundesliga", "Serie A",
                                                "Ligue 1"])

st.sidebar.divider()
st.sidebar.header("Rodzaj Zakładu")
bet_type = st.sidebar.radio("Tryb", ["Zwycięzca (1X2)", "Gole (Over/Under 2.5)"])

# ŁADOWANIE MODELU
with st.spinner(f'Analiza: {selected_league}...'):
    raw_data = load_and_prep_data(selected_league)
    if raw_data.empty: st.error("Brak danych."); st.stop()
    processed_data = add_rolling_features(raw_data)

    train_mode = "1X2" if bet_type == "Zwycięzca (1X2)" else "OU25"
    model, features = train_model(processed_data, prediction_type=train_mode)

# --- SEKCJA AUTOMATYCZNEGO WYBORU MECZU ---
st.sidebar.divider()
st.sidebar.header("Wybór Meczu")

# 1. Pobieramy terminarz
fixtures = get_upcoming_fixtures(selected_league)
match_options = ["Wybierz ręcznie..."]
match_map = {}

if not fixtures.empty:
    for idx, row in fixtures.iterrows():
        date_str = row['Date'].strftime('%d.%m') if pd.notnull(row['Date']) else "??"
        label = f"{date_str} | {row['HomeTeam']} vs {row['AwayTeam']}"
        match_options.append(label)
        match_map[label] = row

selected_fixture = st.sidebar.selectbox("Najbliższe mecze:", match_options)

# Domyślne wartości
default_h_idx, default_a_idx = 0, 1
def_o1, def_ox, def_o2 = 2.0, 3.2, 3.5
def_oo, def_ou = 1.9, 1.9

# Automatyczne uzupełnianie
if selected_fixture != "Wybierz ręcznie...":
    m_data = match_map[selected_fixture]
    teams_list = sorted(raw_data['HomeTeam'].unique())
    try:
        default_h_idx = teams_list.index(m_data['HomeTeam'])
        default_a_idx = teams_list.index(m_data['AwayTeam'])

        if pd.notnull(m_data.get('B365H')): def_o1 = float(m_data['B365H'])
        if pd.notnull(m_data.get('B365D')): def_ox = float(m_data['B365D'])
        if pd.notnull(m_data.get('B365A')): def_o2 = float(m_data['B365A'])
        if pd.notnull(m_data.get('B365>2.5')): def_oo = float(m_data['B365>2.5'])
        if pd.notnull(m_data.get('B365<2.5')): def_ou = float(m_data['B365<2.5'])

        st.sidebar.success(f"Wczytano kursy dla: {m_data['HomeTeam']}")
    except ValueError:
        st.sidebar.warning("Nazwa drużyny w terminarzu różni się od bazy. Wybierz z listy poniżej.")

# Wyświetlanie inputów
teams = sorted(raw_data['HomeTeam'].unique())
home_team = st.sidebar.selectbox("Gospodarz", teams, index=default_h_idx)
away_team = st.sidebar.selectbox("Gość", teams, index=default_a_idx)

if bet_type == "Zwycięzca (1X2)":
    st.sidebar.subheader("Kursy")
    odds_1 = st.sidebar.number_input("1 (Home)", value=def_o1, step=0.05)
    odds_x = st.sidebar.number_input("X (Draw)", value=def_ox, step=0.05)
    odds_2 = st.sidebar.number_input("2 (Away)", value=def_o2, step=0.05)
else:
    st.sidebar.subheader("Kursy")
    odds_over = st.sidebar.number_input("Over 2.5", value=def_oo, step=0.05)
    odds_under = st.sidebar.number_input("Under 2.5", value=def_ou, step=0.05)

st.sidebar.divider()
st.sidebar.header("Kapitał")
bankroll = st.sidebar.number_input("Budżet", value=1000)
kelly_fraction = st.sidebar.select_slider("Ryzyko Kelly", options=[0.05, 0.1, 0.2], value=0.1)

# --- 3. GŁÓWNA ANALIZA ---

if st.sidebar.button("PRZEANALIZUJ MECZ", type="primary"):
    try:
        h_stats = processed_data[processed_data['HomeTeam'] == home_team].iloc[-1]
        a_stats = processed_data[processed_data['AwayTeam'] == away_team].iloc[-1]

        # Base Input
        input_dict = {
            'OddsDiff': [(1 / (odds_1 if bet_type == "Zwycięzca (1X2)" else odds_over)) -
                         (1 / (odds_2 if bet_type == "Zwycięzca (1X2)" else odds_under))],
            'Home_Att': [h_stats['Home_Att']], 'Away_Att': [a_stats['Away_Att']],
            'Home_Def': [h_stats['Home_Def']], 'Away_Def': [a_stats['Away_Def']],
            'Home_Form': [h_stats['Home_Form']], 'Away_Form': [a_stats['Away_Form']],
            'Home_Corners_Avg': [h_stats['Home_Corners_Avg']],
            'Away_Corners_Avg': [a_stats['Away_Corners_Avg']]
        }

        if bet_type == "Zwycięzca (1X2)":
            input_dict['B365H'] = [odds_1]
            input_dict['B365A'] = [odds_2]
            input_dict['B365D'] = [odds_x]
            input_df = pd.DataFrame(input_dict)

            # PREDYKCJA MULTICLASS
            probs = model.predict_proba(input_df)[0]
            p_away, p_draw, p_home = probs[0], probs[1], probs[2]

            outcomes = [
                ("GOSPODARZ (1)", p_home, odds_1),
                ("REMIS (X)", p_draw, odds_x),
                ("GOŚĆ (2)", p_away, odds_2)
            ]

        else:  # Over/Under
            input_dict['B365_O25'] = [odds_over]
            input_dict['B365_U25'] = [odds_under]
            input_df = pd.DataFrame(input_dict)

            p_over = model.predict_proba(input_df)[0][1]
            p_under = 1.0 - p_over

            outcomes = [
                ("OVER 2.5", p_over, odds_over),
                ("UNDER 2.5", p_under, odds_under)
            ]

        # WIZUALIZACJA
        st.subheader(f"Analiza: {home_team} vs {away_team}")

        cols = st.columns(len(outcomes))

        for idx, (label, prob, odd) in enumerate(outcomes):
            with cols[idx]:
                st.markdown(f"### {label}")

                # OBLICZAMY KURS FAIR (1 / Prawdopodobieństwo)
                fair_odd = 1 / prob if prob > 0 else 0

                # ZMIANA: Wyświetlamy % oraz Kurs Fair w nawiasie
                st.metric(
                    label="Prawdopodobieństwo (Fair)",
                    value=f"{prob * 100:.1f}%",
                    delta=f"@{fair_odd:.2f}",  # To wyświetli kurs na zielono/czerwono pod spodem
                    delta_color="off"  # Wyłączamy kolorowanie (żeby było szare/neutralne)
                )

                value = (prob * odd) - 1
                stake = calculate_kelly(prob, odd, bankroll, kelly_fraction)

                st.divider()  # Linia oddzielająca dla czytelności

                # Wyświetlanie Value i Decyzji
                if value > 0.05:
                    st.success(f"📈 VALUE: {value * 100:.1f}%")
                    st.markdown(f"**Bukmacher płaci: {odd:.2f}**")  # Przypomnienie kursu bukmachera
                    if stake > 0:
                        st.markdown(f"💰 Stawka: **{stake:.2f} PLN**")
                elif value > 0:
                    st.warning(f"⚠️ Value: {value * 100:.1f}%")
                    st.markdown(f"Kurs: {odd:.2f}")
                else:
                    st.error(f"Brak Value ({value * 100:.1f}%)")
                    st.markdown(f"Kurs: {odd:.2f}")

        st.divider()
        st.subheader("Statystyki (Średnia z 5 meczów)")
        chart_df = pd.DataFrame({
            'Stat': ['Gole Strzelane', 'Gole Tracone', 'Rożne', 'Punkty (Forma)'],
            home_team: [h_stats['Home_Att'], h_stats['Home_Def'], h_stats['Home_Corners_Avg'], h_stats['Home_Form']],
            away_team: [a_stats['Away_Att'], a_stats['Away_Def'], a_stats['Away_Corners_Avg'], a_stats['Away_Form']]
        }).set_index('Stat')
        st.bar_chart(chart_df)
        st.divider()
        st.subheader("Symulacja Dokładnego Wyniku")

        # Obliczamy xG (Oczekiwane Gole) na ten konkretny mecz
        # Średnia z: Atak Gospodarza vs Obrona Gościa
        xg_home = (h_stats['Home_Att'] + a_stats['Away_Def']) / 2
        xg_away = (a_stats['Away_Att'] + h_stats['Home_Def']) / 2

        col_map, col_stat = st.columns([2, 1])

        with col_map:
            # Rysujemy mapę
            fig = plot_score_heatmap(xg_home, xg_away)
            st.pyplot(fig)

        with col_stat:
            st.info(f"🔢 Przewidywany wynik (xG):")
            st.markdown(f"## {xg_home:.2f} - {xg_away:.2f}")

            # Szukamy wyniku z największą szansą
            best_prob = 0
            best_score = (0, 0)
            for h in range(5):
                for a in range(5):
                    p = poisson.pmf(h, xg_home) * poisson.pmf(a, xg_away)
                    if p > best_prob:
                        best_prob = p
                        best_score = (h, a)

            st.write("Najbardziej realny wynik:")
            st.success(f"⚽ {best_score[0]} - {best_score[1]}")
            st.caption(f"Prawdopodobieństwo: {best_prob * 100:.1f}%")
    except Exception as e:
        st.error(f"Błąd analizy: {e}")