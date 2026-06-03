# src/mappings.py

# FIFA 3-letter code → full name as it appears in the Kaggle results.csv dataset
# Covers all teams that appeared in World Cups 2002-2022

FIFA_CODE_TO_NAME = {
    'ALG': 'Algeria',
    'ANG': 'Angola',
    'ARG': 'Argentina',
    'AUS': 'Australia',
    'BEL': 'Belgium',
    'BIH': 'Bosnia and Herzegovina',
    'BRA': 'Brazil',
    'CAN': 'Canada',
    'CHI': 'Chile',
    'CHN': 'China PR',
    'CIV': 'Ivory Coast',
    'CMR': 'Cameroon',
    'COL': 'Colombia',
    'CRC': 'Costa Rica',
    'CRO': 'Croatia',
    'CZE': 'Czech Republic',
    'DEN': 'Denmark',
    'ECU': 'Ecuador',
    'EGY': 'Egypt',
    'ENG': 'England',
    'ESP': 'Spain',
    'FRA': 'France',
    'GER': 'Germany',
    'GHA': 'Ghana',
    'GRE': 'Greece',
    'HON': 'Honduras',
    'IRL': 'Republic of Ireland',
    'IRN': 'Iran',
    'ISL': 'Iceland',
    'ITA': 'Italy',
    'JPN': 'Japan',
    'KOR': 'South Korea',
    'KSA': 'Saudi Arabia',
    'MAR': 'Morocco',
    'MEX': 'Mexico',
    'NED': 'Netherlands',
    'NGA': 'Nigeria',
    'NZL': 'New Zealand',
    'PAN': 'Panama',
    'PAR': 'Paraguay',
    'PER': 'Peru',
    'POL': 'Poland',
    'POR': 'Portugal',
    'PRK': 'North Korea',
    'QAT': 'Qatar',
    'RSA': 'South Africa',
    'RUS': 'Russia',
    'SEN': 'Senegal',
    'SRB': 'Serbia',
    'SUI': 'Switzerland',
    'SVK': 'Slovakia',
    'SVN': 'Slovenia',
    'SWE': 'Sweden',
    'TOG': 'Togo',
    'TRI': 'Trinidad and Tobago',
    'TUN': 'Tunisia',
    'TUR': 'Turkey',
    'UKR': 'Ukraine',
    'URU': 'Uruguay',
    'USA': 'United States',
    'WAL': 'Wales',
}


def code_to_name(code):
    """
    Translates a FIFA 3-letter code to the full name used in the Kaggle dataset.
    Raises a clear error if the code is missing, so bugs surface immediately
    rather than silently producing wrong results.
    """
    if code not in FIFA_CODE_TO_NAME:
        raise KeyError(f"FIFA code '{code}' not found in mappings. Add it to mappings.py.")
    return FIFA_CODE_TO_NAME[code]