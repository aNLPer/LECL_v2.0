import re

# law内容过滤
def filterStr(law):
    # 删除第一个标点之前的内容
    pattern_head_content = re.compile(r".*?[，：。,:.]")
    head_content = pattern_head_content.match(law)
    if head_content is not None:
        head_content_span = head_content.span()
        law = law[head_content_span[1]:]

    # 删除“讼诉机关认为，......”
    pattern_3 = re.compile(r"[，。]公诉机关")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]+1]

    # 删除"。...事实，"
    pattern_3 = re.compile(r"。.{2,8}事实，")
    content = pattern_3.search(law)
    if content is not None:
        content_span = content.span()
        law = law[:content_span[0]]

    # 删除括号及括号内的内容
    pattern_bracket = re.compile(r"[<《【\[(（〔].*?[〕）)\]】》>]")
    law = pattern_bracket.sub("", law)

    return law


if __name__=="__main__":
    print("\n\tsfaf".strip())

