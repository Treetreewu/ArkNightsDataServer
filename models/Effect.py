import django.db.models as models
import enum
from data_bus.data_loader import loaded_data

TypeSimple = enum.Enum("TypeSimple", ("V", "V1"))


class Type(enum.Enum):
    VALUE_ADD = 1
    VALUE_MULTIPLE = 2
    VALUE_BLOCK = 3
    SILENCE = 4
    STUN = 5
    REPULSE = 6


class ValueTarget(enum.Enum):
    MAX_HP = 1
    HP = 2
    ATK = 3
    DEF = 4
    MAGIC_RESISTANCE = 5
    BLOCK_COUNT = 6
    MOVE_SPEED = 7
    ATTACK_SPEED = 8
    HP_RECOVER_PS = 9
    SP_RECOVER_PS = 10


class Target(enum.Enum):
    SELF = 1
    ALLY = 2
    ENEMY = 3


class Effect(models.Model):
    type = models.IntegerField(choices=[(t.value, t.name) for t in Type])
    value_target = models.IntegerField(choices=[(t.value, t.name) for t in ValueTarget], default=0)
    value = models.DecimalField(max_digits=10, decimal_places=4, help_text="", default=0)
    target = models.SmallIntegerField(choices=[(t.value, t.name) for t in Target])
    count = models.IntegerField(help_text="作用的目标个数, -1=all_in_range", default=1)
    duration = models.DecimalField(max_digits=10, decimal_places=4, help_text="0=瞬间效果，-1=持续时间无限", default=0)
    range = models.CharField()


if __name__ == '__main__':
    pass

