from svgcheck import log

import svgcheck.word_properties as wp
import re

indent = 4
errorCount = 0

trace = True

bad_namespaces = []


def strip_prefix(element, el):
    """
    Given the tag for an element, separate the namespace from the tag
    and return a tuple of the namespace and the local tag
    It will be up to the caller to determine if the namespace is acceptable
    """

    ns = None
    if element[0] == "{":
        rbp = element.rfind("}")  # Index of rightmost }
        if rbp >= 0:
            ns = element[1:rbp]
            element = element[rbp + 1:]
        else:
            log.warn("Malformed namespace.  Should have errored during parsing")
    return element, ns  # return tag, namespace

def value_ok(obj, v):
    log.note("value_ok look for %s in %s" % (v, obj))

    if obj in wp.properties:
        values = wp.properties[obj]
    elif obj in wp.basic_types:
        values = wp.basic_types[obj]
        # Tu peux ajouter ici une vérification ou juste accepter par défaut
        return True, v  # <- Toujours retourner quelque chose
    elif isinstance(obj, str):
        if obj[0] == "+":
            n = re.match(r"\d+\.\d+%?$", v)
            rv = n.group() if n else None
            return True, rv
        if v == obj:
            return True, v
        return False, None
    else:
        return False, None

    # Par défaut, si on a wp.properties ou wp.basic_types mais aucun test
    return True, v
    
def check(el, depth=0):
    global errorCount

    element, ns = strip_prefix(el.tag, el)

    # Vérifie l'élément
    if element not in wp.elements:
        errorCount += 1
        log.warn(f"Element '{element}' not allowed", where=el)
        elementAttributes = []  # Pas d'attributs à vérifier
    else:
        elementAttributes = wp.elements[element]

    for nsAttrib, val in el.attrib.items():
        attr, ns = strip_prefix(nsAttrib, el)

        if attr == "style" and element in ("line", "svg"):
            continue

        if (attr not in elementAttributes) and (attr not in wp.properties):
            errorCount += 1
            log.warn(
                f"The element '{element}' does not allow the attribute '{attr}'", 
                where=el
            )
        elif attr in wp.properties:
            vals = wp.properties[attr]
            ok, new_val = value_ok(attr, val)
            if vals and not ok:
                errorCount += 1

    for child in el:
        if not isinstance(child.tag, str):
            continue
        check(child, depth + 1)

    return errorCount