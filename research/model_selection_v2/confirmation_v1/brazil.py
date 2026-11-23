"""Pinned Brazil input loader. Default inspection never loads the outcome.

No fitting or experiment command is provided. Explicit source lists prevail over
the draft's exclusion-count typo: 54 columns minus 25 exclusions = 29 predictors.
"""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from .schema import Schema

RAW_SHA256 = 'ae00807f566e7c9a4ab804432665a7d50ba54de0d9530f064c8e20f2d32c25c6'
DICTIONARY_SHA256 = 'afb8860352b26da04044b11425fbf4e96c51e71252a2eee39af5cca71a720355'
NUMERIC = tuple('AGE QUANT_DEPENDANTS MONTHS_IN_RESIDENCE PERSONAL_MONTHLY_INCOME OTHER_INCOMES PERSONAL_ASSETS_VALUE QUANT_CARS QUANT_BANKING_ACCOUNTS MONTHS_IN_THE_JOB'.split())
UNORDERED = tuple('APPLICATION_SUBMISSION_TYPE POSTAL_ADDRESS_TYPE MARITAL_STATUS STATE_OF_BIRTH NACIONALITY RESIDENCIAL_STATE FLAG_RESIDENCIAL_PHONE RESIDENCE_TYPE FLAG_EMAIL FLAG_VISA FLAG_MASTERCARD FLAG_DINERS FLAG_AMERICAN_EXPRESS FLAG_OTHER_CARDS COMPANY FLAG_PROFESSIONAL_PHONE PROFESSION_CODE OCCUPATION_TYPE PRODUCT'.split())
EXCLUDED = tuple('ID_CLIENT SEX TARGET_LABEL_BAD=1 CLERK_TYPE QUANT_ADDITIONAL_CARDS EDUCATION_LEVEL FLAG_MOBILE_PHONE FLAG_HOME_ADDRESS_DOCUMENT FLAG_RG FLAG_CPF FLAG_INCOME_PROOF FLAG_ACSP_RECORD CITY_OF_BIRTH RESIDENCIAL_CITY RESIDENCIAL_BOROUGH RESIDENCIAL_PHONE_AREA_CODE RESIDENCIAL_ZIP_3 PROFESSIONAL_ZIP_3 PROFESSIONAL_STATE PROFESSIONAL_CITY PROFESSIONAL_BOROUGH PROFESSIONAL_PHONE_AREA_CODE MATE_PROFESSION_CODE MATE_EDUCATION_LEVEL QUANT_SPECIAL_BANKING_ACCOUNTS'.split())
MISSING = '<MISSING>'


@dataclass
class BrazilInputs:
    predictors: pd.DataFrame
    metadata: pd.DataFrame
    schema: Schema
    label_codes: dict
    source_hashes: dict

    def prepare(self, fitting_positions):
        """Fill every role from this repeat's fitting medians; groups stay raw."""
        positions = np.asarray(fitting_positions)
        if positions.ndim != 1 or not len(positions) or positions.dtype.kind not in 'iu' or len(set(positions)) != len(positions):
            raise ValueError('Explicit unique fitting row positions required')
        if positions.min() < 0 or positions.max() >= len(self.predictors):
            raise ValueError('Fitting row position outside cohort')
        frame = self.predictors.copy(deep=True)
        medians = {}
        for name in self.schema.numeric:
            value = float(frame.iloc[positions][name].median())
            if not np.isfinite(value):
                raise ValueError('Undefined fitting median: '+name)
            medians[name] = value
            frame[name] = frame[name].fillna(value)
        self.schema.validate(frame)
        return frame, medians


def load(root, *, include_outcome=False):
    root = Path(root)
    paths = dict(raw=root/'data/raw/brazil_pakdd2010/PAKDD2010_Modeling_Data.txt',
                 dictionary=root/'data/raw/brazil_pakdd2010/PAKDD2010_VariablesList.XLS')
    hashes = {k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
    if hashes != dict(raw=RAW_SHA256,dictionary=DICTIONARY_SHA256):
        raise ValueError('Brazil source identity failed')
    dictionary = pd.read_excel(paths['dictionary'])
    names = dictionary.Var_Title.tolist()
    if len(names) != 54 or names[43] != 'EDUCATION_LEVEL':
        raise ValueError('Unexpected dictionary layout')
    names[43] = 'MATE_EDUCATION_LEVEL'
    order = tuple(n for n in names if n not in EXCLUDED)
    if len(order) != 29 or set(order) != set(NUMERIC+UNORDERED+('PAYMENT_DAY',)) or set(EXCLUDED) != set(names)-set(order):
        raise ValueError('Explicit inclusion/exclusion lists do not partition dictionary')
    schema = Schema(order,tuple(n for n in order if n in NUMERIC),
        {'PAYMENT_DAY':(1,5,10,15,20,25)},tuple(n for n in order if n in UNORDERED),{'PAYMENT_DAY':(-1,)})
    columns = list(order)+['ID_CLIENT','SEX']+(['TARGET_LABEL_BAD=1'] if include_outcome else [])
    data = pd.read_csv(paths['raw'],sep='\t',header=None,names=names,usecols=columns,
        encoding='latin1',dtype=str,keep_default_na=False)
    if len(data) != 50000:
        raise ValueError('Unexpected cohort size')
    frame = data.loc[:,list(order)].copy()
    labels = {}
    for name in order:
        missing = frame[name].str.strip().isin(['','NULL'])
        if name in schema.numeric:
            frame[name] = pd.to_numeric(frame[name].mask(missing),errors='raise').astype(float)
        elif name in schema.ordered:
            frame[name] = pd.to_numeric(frame[name].mask(missing,'-1'),errors='raise').astype(int)
            if not set(frame[name]) <= set(schema.ordered[name]) | {-1}:
                raise ValueError('Unexpected PAYMENT_DAY code')
        else:
            if frame[name].eq(MISSING).any():
                raise ValueError('Missing sentinel collides with an original label')
            values = frame[name].mask(missing,MISSING)
            labels[name] = {label:i for i,label in enumerate(sorted(values.unique()))}
            frame[name] = values.map(labels[name]).astype(int)
    # Group before fitting-median fill. Coding is a bijection, so membership is
    # identical to original labels; missing values match other missing values.
    groups = frame.groupby(list(order),dropna=False,sort=False).ngroup()
    sex = data.SEX.str.strip()
    if not set(sex) <= {'F','M','N',''}:
        raise ValueError('Unexpected recorded-sex code')
    meta = pd.DataFrame(dict(ID=pd.to_numeric(data.ID_CLIENT,errors='raise'),
        SEX=sex.map({'F':2,'M':1,'N':0,'':0}),duplicate_group_id=groups))
    if not meta.ID.is_unique or meta.isna().any().any():
        raise ValueError('Invalid metadata')
    if include_outcome:
        meta['default'] = pd.to_numeric(data['TARGET_LABEL_BAD=1'],errors='raise').astype(int)
        if not meta.default.isin([0,1]).all():
            raise ValueError('Invalid binary outcome')
    return BrazilInputs(frame,meta,schema,labels,hashes)
