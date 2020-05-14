import django.db.models as models
import enum

TypeSimple = enum.Enum("TypeSimple", ("V", "V1"))


class Type(enum.Enum):
    VALUE_ADD = 1
    VALUE_MULTIPLE = 2
    VALUE_BLOCK = 3
    SILENCE = 4
    STUN = 5
    REPULSE = 6


class Target(enum.Enum):
    SELF = 1
    ALLY = 2
    ENEMY = 3


class Range(enum.Enum):
    """
    Typical value.
    """
    MONO_CASTER_0 = (((-1, 0), (1, 2)),) # 口
    MONO_CASTER_1 = (((-1, 0), (1, 2)), (0, 3)) # 凸

    MULTI_CASTER_0 = (((-1, 0), (1, 1)), (0, 2)) # 凸
    MULTI_CASTER_1 = MONO_CASTER_0
    PIONEER = (((0, 0), (0, 1)),)
    WARRIOR = PIONEER
    MONO_MEDIC_0 = MONO_CASTER_1
    MONO_MEDIC_1 = (((-1, 0), (1, 3)),)


class Effect(models.Model):
    type = models.IntegerField(choices=[(t.value, t.name) for t in Type])
    target = models.SmallIntegerField(choices=[(t.value, t.name) for t in Target])
    count = models.IntegerField(help_text="作用的目标个数")
    duration = models.IntegerField(help_text="s")


if __name__ == '__main__':
    pass

