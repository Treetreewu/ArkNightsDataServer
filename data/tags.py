
BUFFS = [buff+'buff' for buff in ['输出', '术士', '先锋', '狙击', '近战', '物抗', '攻速', '三星', '物理闪避', '法术闪避', '生命回复', '韧性']]
DEBUFFS = [debuff+'debuff' for debuff in ['输出', '重量', '物抗', '法抗']]
TAGS = ['治疗', '爆发', '控制', '位移', 'AOE', '攻击范围', '远程', '近战', '技力回复', '空射', '优先目标'
        '物理输出', '法术输出', '真实伤害', '生命回复', '物理闪避', '法术闪避', '沉默', '撤离回费', '换形师', '召唤师'] + BUFFS + DEBUFFS
# 近战指攻击范围为 ◻ 或 ◻◻，其他均视为远程。
# 为了偷懒，这里将抵抗和格挡视作闪避。

# 可省略，自动填充的标签：
#   医疗干员 的 治疗
#   术士 辅助干员 的 法术输出
#   狙击 近卫干员 的 物理输出
# # 注：对于术士 辅助 狙击 近卫干员：仅当物理输出、法术输出标签都缺失时，才会自动填充。


wifi_tags = {
    "凛冬": ['先锋buff'],
    "幽灵鲨": ['AOE', '生命回复', '物理输出'],
    "杜宾": ['三星buff', '远程'],
    "推进之王": ['AOE', '先锋buff', '控制', '物理输出'],
    "白面鸮": ['技力回复', '爆发'],
    "芬": [],
    "深海色": ['召唤师', '物理闪避buff'],
    "德克萨斯": ['控制', '物理输出'],
    "夜刀": ['快速复活'],
    "芙兰卡": ['真实伤害', '物理输出'],
    "巡林者": ['空射', '物理输出'],
    "流星": ['物抗debuff', '空射', '物理输出'],
    "杜林": ['法术闪避'],
    "克洛丝": ['物理输出'],
    "夜烟": ['法抗debuff'],
    "炎熔": ['AOE'],
    "梅": ['控制', '物理输出'],
    "白雪": ['攻击范围', '控制', 'AOE', '法术输出', '物理输出'],
    "普罗旺斯": [],
    "阿米娅": ['攻击范围', '法术输出', '真实伤害', 'AOE'],
    "远山": ['攻击范围', 'AOE'],
    "能天使": ['物理输出', '爆发'],
    "蓝毒": ['AOE', '物理输出'],
    "雷蛇": ['技力回复', '远程', '控制', 'AOE'],
    "星熊": ['物理闪避', '法术闪避', '重装buff', 'AOE'],
    "红": ['控制', '物理闪避'],
    "闪灵": ['物抗buff'],
    "末药": ['AOE'],
    "赫默": ['AOE', '医疗buff'],
    "蛇屠箱": ['生命回复'],
    "米格鲁": [],
    "芙蓉": [],
    "临光": ['治疗', '医疗buff'],
    "伊芙利特": ['攻击范围', 'AOE', '法抗debuff', '物抗debuff', '法术输出'],
    "银灰": ['远程', 'AOE', '生命回复'],
    "塞雷娅": ['AOE', '治疗', '技力回复', '控制', '法抗debuff'],
    "夜魔": ['治疗', '物理闪避', '法术闪避', '控制'],
    "天火": ['AOE', '控制'],
    "因陀罗": ['物理闪避', '法术输出', '物理输出', '生命回复'],
    "初雪": ['AOE', '控制', '攻击范围', '法抗debuff', '物抗debuff'],
    "艾丝黛尔": ['AOE', '生命回复'],
    "猎蜂": ['物理闪避', ],
    "嘉维尔": ['医疗buff', 'AOE'],
    "卡缇": ['生命回复'],
    "史都华德": ['优先目标'],
    "艾雅法拉": ['攻击范围', 'AOE', '爆发', '法抗debuff'],
    "夜莺": ['AOE', '攻击范围', '法抗buff', '召唤师', '法术闪避buff'],
    "陈": ['爆发', '控制', 'AOE', '法术输出', '物理输出'],
    "拉普兰德": ['远程', '法术输出', '物理输出', '沉默'],
    "华法琳": ['输出buff', '技力回复'],
    "守林人": ['爆发', '攻击范围', 'AOE'],
    "狮蝎": ['攻击范围', 'AOE', '控制', '远程'],
    "火神": ['AOE', '换形师', '生命回复'],
    "真理": ['控制', '法术输出'],
    "慕斯": ['输出debuff', '法术输出'],
    "暗索": ['位移', '远程', '物理闪避'],
    "砾": [],
    "凯尔希": ['召唤师', '攻击范围', '物理输出', '法术输出', 'AOE'],
    "地灵": ['控制'],
    "调香师": ['生命回复buff'],
    "讯使": [],
    "陨星": ['AOE', '物抗debuff'],
    "白金": ['攻击范围'],
    "香草": [],
    "可颂": ['位移', '物理闪避', '法术闪避', '物理闪避buff', '法术闪避buff'],
    "梅尔": ['AOE', '爆发', '召唤师', '输出debuff', '控制'],
    "崖心": ['远程', '位移', '控制', '真实伤害'],
    "空": ['生命回复buff', 'aoe', '控制', '输出buff'],
    "食铁兽": ['AOE', '位移', '物理闪避'],
    "霜叶": ['控制', '远程'],
    "清道夫": [],
    "古米": ['治疗', '控制'],
    "角峰": [],
    "断罪者": [],
    "玫兰莎": ['物理输出'],
    "安赛尔": ['治疗', '攻击范围'],
    "缠丸": ['生命回复', '物理输出'],
    "阿消": ['位移'],
    "安洁莉娜": ['输出buff', '生命回复buff', '重量debuff', 'AOE', '法术输出'],
    "红豆": ['物理输出'],
    "暴行": ['AOE', '爆发', '物理输出'],
    "杰西卡": ['物理输出', '物理闪避', '法术闪避'],
    "12F": ['物理闪避', 'AOE'],
    "梓兰": ['减速'],
    "翎羽": [],
    "Lancet-2": ['治疗'],
    "Castle-3": ['近战buff'],
    "安德切尔": ['优先目标'],
    "空爆": ['AOE', '物理输出'],
    "斯卡蒂": ['深海猎人buff', '物理输出'],
    "格拉尼": ['换形师', '先锋buff', '物理闪避', 'AOE'],
    "月见夜": ['物理输出', '法术输出', '远程'],
    "斑点": ['治疗', '物理闪避buff'],
    "泡普卡": ['AOE'],
    "格雷伊": ['AOE', '法术输出', '控制'],
    "诗怀雅": ['远程', '近战buff'],
    "苏苏洛": ['治疗'],
    "格劳克斯": ['优先目标', '控制', '法术输出', 'AOE'],
    "锡兰": ['攻击范围', '治疗', '韧性buff'],
    "黑角": [],
    "黑": ['狙击buff', '物抗debuff'],
    "桃金娘": ['治疗', '生命回复buff'],
    "星极": ['法术输出', 'AOE'],
    "赫拉格": ['生命回复', '物理闪避', '攻击范围', 'AOE', '物理输出']
}

tags = []
if "位移" in tags:
    tags.append("控制")


