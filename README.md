# Car evalution-Theme&emotion

**赛题名称**

汽车行业用户观点主题及情感识别（CCF 2018）

**赛题背景:**
	
随着政府对新能源汽车的大力扶植以及智能联网汽车兴起都预示着未来几年汽车行业的多元化发展及转变。汽车厂商需要了解自身产品是否能够满足消费者的需求，但传统的调研手段因为样本量小、效率低等缺陷已经无法满足当前快速发展的市场环境。因此，汽车厂商需要一种快速、准确的方式来了解消费者需求。

**任务描述：**
	
本赛题提供一部分网络中公开的用户对汽车的相关内容文本数据作为训练集，训练集数据已由人工进行分类并进行标记，参赛队伍需要对文本内容中的讨论主题和情感信息来分析评论用户对所讨论主题的偏好。讨论主题可以从文本中匹配，也可能需要根据上下文提炼。
	
**数据背景：**
	
	数据为用户在汽车论坛中对汽车相关内容的讨论或评价。

**数据说明：**

（1）训练数据： 训练数据为CSV格式，以英文半角逗号分隔，首行为表头。

· 训练集数据中主题被分为10类，包括：动力、价格、内饰、配置、安全性、外观、操控、油耗、空间、舒适性。

· 情感分为3类，分别用数字0、1、-1表示中立、正向、负向。

· content_id与content一一对应，但同一条content中可能会包含多个主题，此时出现多条记录标注不同的主题及情感，因此在整个训练集中content_id存在重复值。其中content_id，content，subject，sentiment_value对应字段不能为空且顺序不可更改，否则提交失败。

· 仅小部分训练数据包含有情感词sentiment_word，大部分为空，情感词不作为评分依据。

	字段顺序为：content_id，content，subject，sentiment_value，sentiment_word

（2）测试数据：测试数据为CSV格式，首行为表头，字段为：
		
	content_id，content
      

**代码说明：**
    
  included：
    
    Pre-trained word2vector, 
    
    shuffle————increase the amount of data,
    
    dictionary generalization, including car brand, location, time, money, etc.
    
    Multi-target recognition;
   
  code file:

  	main file: You can train and predict by running this file.
  	util file: This file customizes a number of called functions,including data processing functions, vocabulary generalization functions, etc.
  	model file: This file is used to configure the network architecture.
  	F1-socre file: This file is used to define F1-score, which is calculated according to the requirements of the competition.
  	val file: This file is used to convert the validation set file.
