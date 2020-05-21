import pandas as pd
import numpy as np
from scipy import interpolate
import attr
from typing import Dict
from typing import List
from typing import Tuple
from datetime import date

MitigationStrategy_Type = List[Tuple[date, float]]

AGE_GROUPS = [
    '0-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70-79',
    '80+'
]

NUM_AGE_GROUPS = len(AGE_GROUPS)

# age_specific_hospitalization_rates = {
#     '0-9': 0.01,
#     '10-19': 0.03,
#     '20-29': 0.03,
#     '30-39': 0.03,
#     '40-49': 0.06,
#     '50-59': 0.10,
#     '60-69': 0.25,
#     '70-79': 0.35,
#     '80+': 0.50
# }

# age_specific_icu_rates = {
#     '0-9': 0.05,
#     '10-19': 0.10,
#     '20-29': 0.10,
#     '30-39': 0.15,
#     '40-49': 0.20,
#     '50-59': 0.25,
#     '60-69': 0.35,
#     '70-79': 0.45,
#     '80+': 0.55
# }

# percentage of those in icu
# age_specific_fatality_rates = {
#     '0-9': 0.05,
#     '10-19': 0.10,
#     '20-29': 0.10,
#     '30-39': 0.15,
#     '40-49': 0.20,
#     '50-59': 0.25,
#     '60-69': 0.35,
#     '70-79': 0.45,
#     '80+': 0.55
# }

INFECTIOUS_PERIOD_DAYS = 14.0  # FIXME: sources!
INCUBATION_TIME_DAYS = 2.0  # FIXME: sources!
HOSPITAL_STAY_DAYS = 8.0  # ditto
ICU_STAY_DAYS = 21.0  # ditto
OVERFLOW_SEVERITY = 3.0  # ???
AVERAGE_R0 = 6.30

age_specific_rates = pd.DataFrame(
    columns=[
        'Hospitalization',
        'ICU',
        'Fatality',
        'RecoveryRate',
        'HospitalizationRate',
        'DischargeRate',
        'CriticalRate',
        'StabilizationRate',
        'DeathRate',
        'OverflowDeathRate'  # if no ECMO/ventilator available
    ],
    index=AGE_GROUPS
)


class EpidemicState(object):
    def __init__(self):
        self.susceptible = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.exposed = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.infectious = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.hospitalized = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.critical = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.overflow = np.zeros(shape=(NUM_AGE_GROUPS,))

        self.recovered = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.intensive = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.discharged = np.zeros(shape=(NUM_AGE_GROUPS,))
        self.dead = np.zeros(shape=(NUM_AGE_GROUPS,))


@attr.s
class TravelFluxByAge(object):
    """
    Переезжающие из региона в регион индивиды, в разбивке по возрастам
    """
    susceptible = attr.ib(type=np.ndarray, default=np.zeros(shape=(NUM_AGE_GROUPS,)))
    exposed = attr.ib(type=np.ndarray, default=np.zeros(shape=(NUM_AGE_GROUPS,)))
    # infectious = attr.ib(type=np.ndarray, default=np.zeros(shape=(NUM_AGE_GROUPS,)))
    recovered = attr.ib(type=np.ndarray, default=np.zeros(shape=(NUM_AGE_GROUPS,)))


@attr.s
class TravelFlux(object):
    """
    Переезжающие из региона в регион индивиды - без разбивки по возрастам
    """
    susceptible: float = 0.0
    exposed: float = 0.0
    # infectious: float = 0.0
    recovered: float = 0.0


def convert_flux_to_flux_by_age(flux: TravelFlux,
                                age_distribution: np.ndarray) -> TravelFluxByAge:

    # age_distribution can be anything - counts or ratios, we don't care
    ratios = age_distribution / float(np.sum(age_distribution))
    result = TravelFluxByAge()
    result.susceptible = flux.susceptible * ratios
    result.exposed = flux.exposed * ratios
    # result.infectious = flux.infectious * ratios
    result.recovered = flux.recovered * ratios
    return result


@attr.s
class EpidemicParams(object):
    """
    Параметры модифицированной SEIR модели эпидемии

    infectious_period_days - Число дней, в течение которых инфицированный заражает других
    incubation_time_days - Продолжительность инкубационного (латентного) периода, дней
    hospital_stay_days - Среднее время пребывания в обычной палате стационара, дней
    icu_stay_days - Среднее число дней, которое пациенты находятся на ИВЛ в реанимации
    overflow_severity - Пациент в критическом состоянии, не попавший в реанимацию,
                        имеет вероятность умереть в день во столько раз выше,
                        чем пациент на ИВЛ
    average_r0 - Среднее базовое репродуктивное число R_0
    """

    infectious_period_days = attr.ib(default=14.0, type=float, kw_only=True)  # FIXME: sources!
    incubation_time_days = attr.ib(default=2.0, type=float, kw_only=True)  # FIXME: sources!
    hospital_stay_days = attr.ib(default=8.0, type=float, kw_only=True)  # ditto
    icu_stay_days = attr.ib(default=21.0, type=float, kw_only=True)  # ditto
    overflow_severity = attr.ib(default=3.0, type=float, kw_only=True)  # ???
    average_r0 = attr.ib(default=3.0, type=float, kw_only=True)
    mitigation_strategy = attr.ib(type=MitigationStrategy_Type,
                                  kw_only=True,
                                  default=None)

    simulation_start_date = attr.ib(default=date(2020, 1, 1), type=date, kw_only=True)
    simulation_days = attr.ib(default=366, type=int, kw_only=True)

    """
    Совокупная вероятность госпитализации для зараженного индивида
    """
    hospitalization_probabilities = attr.ib(
        type=Dict[str, float],
        default={
            '0-9':   0.01,
            '10-19': 0.03,
            '20-29': 0.03,
            '30-39': 0.03,
            '40-49': 0.06,
            '50-59': 0.10,
            '60-69': 0.25,
            '70-79': 0.35,
            '80+':   0.50
        },
        kw_only=True)

    """
    Совокупная вероятность перехода из стационара в реанимацию
    """
    icu_probabilities = attr.ib(
        type=Dict[str, float],
        default={
            '0-9':   0.05,
            '10-19': 0.10,
            '20-29': 0.10,
            '30-39': 0.15,
            '40-49': 0.20,
            '50-59': 0.25,
            '60-69': 0.35,
            '70-79': 0.45,
            '80+':   0.55
        },
        kw_only=True
    )

    """
    Совокупная вероятность смерти для пациентов, нуждающихся в реанимации
    """
    fatality_probabilities = attr.ib(
        type=Dict[str, float],
        default={
            '0-9':   0.05,
            '10-19': 0.10,
            '20-29': 0.10,
            '30-39': 0.15,
            '40-49': 0.20,
            '50-59': 0.25,
            '60-69': 0.35,
            '70-79': 0.45,
            '80+':   0.55
        },
        kw_only=True
    )

    regions: List[str] = ['Россия']

    def __attrs_post_init__(self):

        self.recovery_rates: Dict[str, float] = {}
        self.hospitalization_rates: Dict[str, float] = {}
        self.discharge_rates: Dict[str, float] = {}
        self.critical_rates: Dict[str, float] = {}
        self.stabilization_rates: Dict[str, float] = {}
        self.death_rates: Dict[str, float] = {}
        self.overflow_death_rates: Dict[str, float] = {}
        self.overflow_stabilization_rates: Dict[str, float] = {}
        self.dates = pd.date_range(self.simulation_start_date,
                                   periods=self.simulation_days)

        # TODO: allow for multiple R0 trajectories
        # ndx_regions_and_dates = pd.MultiIndex.from_product([self.dates, self.regions],
        #                                                    names=['regions', 'dates'])

        if self.mitigation_strategy is None:
            self.r0_trajectory = pd.Series(data=self.average_r0,
                                           index=self.dates)
        else:
            mitigation_dates = np.array([x.toordinal() for x in [value[0] for value in self.mitigation_strategy]])
            mitigation_strictness = np.array([value[1] for value in self.mitigation_strategy])
            mitigation_function = interpolate.interp1d(mitigation_dates, mitigation_strictness, kind='linear')
            self.r0_trajectory = pd.Series([mitigation_function(d.toordinal()) * self.average_r0 for d in self.dates],
                                           index=self.dates)

        for i in AGE_GROUPS:
            self.recovery_rates[i] = (1.0 - self.hospitalization_probabilities[i]) \
                / float(self.infectious_period_days)
            self.hospitalization_rates[i] = self.hospitalization_probabilities[i] \
                / float(self.infectious_period_days)
            self.discharge_rates[i] = (1.0 - self.icu_probabilities[i]) \
                / float(self.hospital_stay_days)
            self.critical_rates[i] = self.icu_probabilities[i] \
                / float(self.hospital_stay_days)
            self.stabilization_rates[i] = (1.0 - self.fatality_probabilities[i]) \
                / float(self.icu_stay_days)
            self.death_rates[i] = self.fatality_probabilities[i] \
                / float(self.icu_stay_days)
            self.overflow_stabilization_rates[i] = (1.0 / self.overflow_severity) * self.stabilization_rates[i]
            self.overflow_death_rates[i] = min(1.0,
                                               self.overflow_severity * self.death_rates[i])
            # assert(self.death_rates[i] <= self.overflow_death_rates[i])
