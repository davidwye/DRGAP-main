from enum import Enum


class Gender(Enum):
  UNKNOWN = 0
  MASCULINE = 1
  FEMININE = 2


PRONOUNS = {
    'she': Gender.FEMININE,
    'her': Gender.FEMININE,
    'hers': Gender.FEMININE,
    'he': Gender.MASCULINE,
    'his': Gender.MASCULINE,
    'him': Gender.MASCULINE,
}


class Stereotype(Enum):
    ANTISTEREOTYPE = -1
    NEUTRAL = 0
    STEREOTYPE = 1


STEREOTYPE = {
    -1: Stereotype.ANTISTEREOTYPE,
    0: Stereotype.NEUTRAL,
    1: Stereotype.STEREOTYPE,
}

