def 套娃(i):
    if i < 100:
        return "禁止" + str(套娃(i + 1))

print(套娃(0) + "套娃")