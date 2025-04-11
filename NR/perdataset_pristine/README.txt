We give the k-fold pristine model for all datasets we used in our paper.
To test, this is the corresponding:
    for SJTU: {1: ['hhi'],2: ['longdress'],3: ['loot'],4: ['redandblack'],5: ['Romanoillamp'],
                    6: ['shiva'],7: ['soldier'],8: ['statue'],9: ['ULB_Unicorn']}
    for WPC: {1:["banana", "cauliflower", "mushroom", "pineapple"],
              2:["bag", "biscuits", "cake", "flowerpot"],
              3:["glasses_case", "honeydew_melon", "house", "pumpkin"],
              4:["litchi", "pen_container", "ping-pong_bat", "puer_tea"],
              5:["ship", "statue", "stone", "tool_box"]}
    for ICIP: {1: ['longdress'],2: ['loot'],3: ['redandblack'],4: ['ricardo'], 5: ['sarah9'],6: ['soldier']}
    for LS-PCQA: we give the mos_all.csv and mos_sel.csv respectively have 930 and 840(no_local) point clouds
                the 5 fold is divided by index [0,185,375,560,745,930] and [0,170,350,510,670,840]
    note: the above k-fold partition is based on AFQ-Net(https://github.com/zhangyujie-1998/AFQ-Net)
