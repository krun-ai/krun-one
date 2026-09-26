# Error analysis — krun_one_B (val_eval only)

Abstention score `top1_top2_margin`, operating point 0.98. Rates = errors / decisions of that slice. Baseline column: `laya` on the same decisions.

## kb_val

| category | slice | errors / n | rate | baseline rate |
|---|---|---:|---:|---:|
| option_count | intent:k16 | 86 / 1221 | 7.0% | 43.8% |
| option_count | intent:k2 | 28 / 1218 | 2.3% | 25.0% |
| option_count | intent:k4 | 37 / 1290 | 2.9% | 29.3% |
| option_count | intent:k8 | 67 / 1341 | 5.0% | 36.8% |
| multilingual | en | 58 / 1368 | 4.2% | 22.1% |
| multilingual | en-US | 34 / 1234 | 2.8% | 25.4% |
| multilingual | es-ES | 65 / 1234 | 5.3% | 44.2% |
| multilingual | pt-PT | 61 / 1234 | 4.9% | 44.5% |
| ood_false_accept | ood_cross_dataset | 2 / 150 | 1.3% | 0.0% |
| ood_false_accept | ood_label_removed | 54 / 544 | 9.9% | 0.0% |
| permutation_flip | groups_disagreeing | 2 / 183 | 1.1% | 32.2% |
| semantic_confusion (count of errors) | intent:banking77 | 18 | | |
| semantic_confusion (count of errors) | intent:hwu64 | 40 | | |
| semantic_confusion (count of errors) | intent:massive | 160 | | |
| ambiguous (count of errors) | intent | 36 | | |

Top intent confusions (gold -> predicted): `qa_maths -> qa_factoid` ×9; `recommendation_events -> recommendation_movies` ×8; `play_music -> play_audiobook` ×7; `calendar_set -> calendar_query` ×6; `takeaway order -> takeaway query` ×6; `general_greet -> general_quirky` ×5; `general affirm -> general confirm` ×5; `play_music -> qa_factoid` ×4; `recommendation_events -> calendar_query` ×4; `general explain -> general repeat` ×4

<details><summary>examples wrong:intent</summary>

```json
[
 {
  "context": "How long does it take for a new card to ship?",
  "gold": "card_arrival",
  "pred": "card_delivery_estimate",
  "k": 16,
  "p_pred": 0.967,
  "language": "en"
 },
 {
  "context": "How long does it take for me to get my new card?",
  "gold": "card_arrival",
  "pred": "card_delivery_estimate",
  "k": 16,
  "p_pred": 0.603,
  "language": "en"
 },
 {
  "context": "Am I charged if I make a top-up?",
  "gold": "top_up_by_bank_transfer_charge",
  "pred": "top_up_by_card_charge",
  "k": 2,
  "p_pred": 0.993,
  "language": "en"
 },
 {
  "context": "why isn't my top-up going through?",
  "gold": "pending_top_up",
  "pred": "top_up_reverted",
  "k": 4,
  "p_pred": 0.525,
  "language": "en"
 },
 {
  "context": "How can I get my money that I didn't get from the ATM?",
  "gold": "wrong_amount_of_cash_received",
  "pred": "cash_withdrawal_not_recognised",
  "k": 16,
  "p_pred": 0.81,
  "language": "en"
 },
 {
  "context": "i disagree with a charge",
  "gold": "card_payment_not_recognised",
  "pred": "direct_debit_payment_not_recognised",
  "k": 4,
  "p_pred": 0.658,
  "language": "en"
 },
 {
  "context": "I cannot tell what my top up currently is.",
  "gold": "topping_up_by_card",
  "pred": "pending_top_up",
  "k": 16,
  "p_pred": 0.424,
  "language": "en"
 },
 {
  "context": "Purchases I did not make are appearing in my bank statement?",
  "gold": "direct_debit_payment_not_recognised",
  "pred": "card_payment_not_recognised",
  "k": 4,
  "p_pred": 0.73,
  "language": "en"
 }
]
```
</details>

<details><summary>examples ood:ood_label_removed</summary>

```json
[
 {
  "context": "When will I receive my card?",
  "accepted": "card_arrival",
  "conf": 0.999,
  "options": [
   "verify_source_of_funds",
   "unable_to_verify_identity",
   "visa_or_mastercard",
   "card_arrival",
   "top_up_failed",
   "Refund_not_showing_up"
  ]
 },
 {
  "context": "My card gets rejected when I try to use it",
  "accepted": "contactless_not_working",
  "conf": 0.955,
  "options": [
   "virtual_card_not_working",
   "contactless_not_working"
  ]
 },
 {
  "context": "Where can I get my currencies exchanged?",
  "accepted": "exchange_rate",
  "conf": 0.962,
  "options": [
   "request_refund",
   "exchange_rate",
   "pending_card_payment",
   "card_not_working"
  ]
 },
 {
  "context": "There are some charges that I don't recognize.  It happened days ago.  Can I get these charges back?  Should I cancel my card?",
  "accepted": "direct_debit_payment_not_recognised",
  "conf": 0.971,
  "options": [
   "Refund_not_showing_up",
   "compromised_card",
   "declined_card_payment",
   "balance_not_updated_after_cheque_or_cash_deposit",
   "card_not_working",
   "virtual_card_not_working"
  ]
 }
]
```
</details>

<details><summary>examples ood:ood_cross_dataset</summary>

```json
[
 {
  "context": "if it's ten p. m. in california what time is it in alabama",
  "accepted": "exchange_via_app",
  "conf": 0.999,
  "options": [
   "declined_cash_withdrawal",
   "compromised_card",
   "wrong_exchange_rate_for_cash_withdrawal",
   "card_payment_wrong_exchange_rate",
   "Refund_not_showing_up",
   "disposable_card_limits"
  ]
 },
 {
  "context": "se são vinte e duas horas na califórnia que horas são no alabama",
  "accepted": "exchange_via_app",
  "conf": 0.917,
  "options": [
   "declined_cash_withdrawal",
   "compromised_card",
   "wrong_exchange_rate_for_cash_withdrawal",
   "card_payment_wrong_exchange_rate",
   "Refund_not_showing_up",
   "disposable_card_limits"
  ]
 }
]
```
</details>

## val_hard

| category | slice | errors / n | rate | baseline rate |
|---|---|---:|---:|---:|
| option_count | intent:k16 | 73 / 849 | 8.6% | 54.5% |
| option_count | intent:k32 | 105 / 804 | 13.1% | 57.2% |
| option_count | intent:k4 | 49 / 806 | 6.1% | 43.8% |
| option_count | intent:k8 | 57 / 803 | 7.1% | 46.5% |
| multilingual | en | 34 / 520 | 6.5% | 38.5% |
| multilingual | en-US | 78 / 914 | 8.5% | 38.8% |
| multilingual | es-ES | 87 / 914 | 9.5% | 58.8% |
| multilingual | pt-PT | 85 / 914 | 9.3% | 60.9% |
| ood_false_accept | cross_task | 1 / 152 | 0.7% | 0.0% |
| ood_false_accept | label_removed | 48 / 473 | 10.1% | 0.0% |
| permutation_flip | groups_disagreeing | 4 / 164 | 2.4% | 52.4% |
| semantic_confusion (count of errors) | intent:banking77 | 34 | | |
| semantic_confusion (count of errors) | intent:massive | 250 | | |
| ambiguous (count of errors) | intent | 34 | | |

Top intent confusions (gold -> predicted): `general_quirky -> weather_query` ×11; `takeaway_order -> takeaway_query` ×9; `calendar_set -> calendar_query` ×9; `general_quirky -> calendar_query` ×7; `qa_factoid -> general_quirky` ×6; `general_quirky -> news_query` ×6; `general_quirky -> qa_factoid` ×6; `play_radio -> play_music` ×6; `general_greet -> general_quirky` ×5; `general_quirky -> qa_definition` ×5

<details><summary>examples wrong:intent</summary>

```json
[
 {
  "context": "How long does it take for a new card to ship?",
  "gold": "card_arrival",
  "pred": "card_delivery_estimate",
  "k": 8,
  "p_pred": 0.869,
  "language": "en"
 },
 {
  "context": "If I find a card I lost do I need to dispose of it? Or can I re-active the card and continue to use it?",
  "gold": "card_linking",
  "pred": "lost_or_stolen_card",
  "k": 8,
  "p_pred": 0.397,
  "language": "en"
 },
 {
  "context": "I would like to re-activate my card, it was previously reported  lost but I found it this morning.",
  "gold": "card_linking",
  "pred": "activate_my_card",
  "k": 16,
  "p_pred": 0.994,
  "language": "en"
 },
 {
  "context": "Which flat currencies can I use from you?",
  "gold": "fiat_currency_support",
  "pred": "supported_cards_and_currencies",
  "k": 16,
  "p_pred": 0.706,
  "language": "en"
 },
 {
  "context": "Is there a way to make my money usable in another country?",
  "gold": "exchange_via_app",
  "pred": "fiat_currency_support",
  "k": 16,
  "p_pred": 0.549,
  "language": "en"
 },
 {
  "context": "Can you send me a new card and block my current one. I'm in Spain and my wallet was stolen.",
  "gold": "lost_or_stolen_card",
  "pred": "compromised_card",
  "k": 32,
  "p_pred": 0.691,
  "language": "en"
 },
 {
  "context": "Am I charged if I make a top-up?",
  "gold": "top_up_by_bank_transfer_charge",
  "pred": "top_up_by_card_charge",
  "k": 8,
  "p_pred": 0.611,
  "language": "en"
 },
 {
  "context": "My account is  new  why can I  not top up",
  "gold": "pending_top_up",
  "pred": "top_up_failed",
  "k": 4,
  "p_pred": 0.858,
  "language": "en"
 }
]
```
</details>

<details><summary>examples ood:label_removed</summary>

```json
[
 {
  "context": "How long does it take for a new card to ship?",
  "accepted": "card_delivery_estimate",
  "conf": 0.898,
  "options": [
   "activate_my_card",
   "order_physical_card",
   "card_delivery_estimate",
   "card_about_to_expire"
  ]
 },
 {
  "context": "Can I choose when my card is delivered?",
  "accepted": "card_arrival",
  "conf": 0.987,
  "options": [
   "getting_spare_card",
   "card_acceptance",
   "lost_or_stolen_card",
   "card_arrival",
   "top_up_by_card_charge",
   "card_about_to_expire"
  ]
 },
 {
  "context": "What do I do if my card was stolen.",
  "accepted": "lost_or_stolen_phone",
  "conf": 0.847,
  "options": [
   "card_swallowed",
   "card_about_to_expire",
   "getting_spare_card",
   "disposable_card_limits",
   "visa_or_mastercard",
   "declined_cash_withdrawal"
  ]
 },
 {
  "context": "Should i uninstall the app before i try it again?",
  "accepted": "lost_or_stolen_phone",
  "conf": 0.834,
  "options": [
   "passcode_forgotten",
   "cash_withdrawal_not_recognised",
   "declined_cash_withdrawal",
   "Refund_not_showing_up",
   "direct_debit_payment_not_recognised",
   "card_not_working"
  ]
 }
]
```
</details>

<details><summary>examples ood:cross_task</summary>

```json
[
 {
  "context": "puedes mostrar una receta de espaguetis",
  "accepted": "is_perfect_square",
  "conf": 0.867,
  "options": [
   "is_perfect_square",
   "Get All Motivational Pictures",
   "DynamicScaleVisualizer",
   "user_follower_list",
   "robotic_arm_operation_schedule",
   "getCustomerServiceNumber"
  ]
 }
]
```
</details>

## val_tools

| category | slice | errors / n | rate | baseline rate |
|---|---|---:|---:|---:|
| option_count | tool:k16 | 15 / 197 | 7.6% | 45.2% |
| option_count | tool:k2 | 4 / 235 | 1.7% | 8.9% |
| option_count | tool:k32 | 21 / 229 | 9.2% | 60.3% |
| option_count | tool:k4 | 5 / 220 | 2.3% | 23.6% |
| option_count | tool:k8 | 12 / 227 | 5.3% | 30.8% |
| tool_slices | description_dependent | 19 / 291 | 6.5% | 47.8% |
| tool_slices | name_matches_query | 38 / 817 | 4.7% | 28.3% |
| tool_slices | near_duplicate_distractor | 23 / 279 | 8.2% | 63.1% |
| tool_slices | no_near_duplicate | 34 / 829 | 4.1% | 23.4% |
| tool_slices | seen_tool | 16 / 527 | 3.0% | 28.8% |
| tool_slices | toolset_large(k>=16) | 36 / 426 | 8.5% | 53.3% |
| tool_slices | toolset_medium(5-15) | 12 / 227 | 5.3% | 30.8% |
| tool_slices | toolset_small(<=4) | 9 / 455 | 2.0% | 16.0% |
| tool_slices | unseen_tool | 41 / 581 | 7.1% | 37.5% |
| ood_false_accept | cross_task | 0 / 47 | 0.0% | 0.0% |
| ood_false_accept | native_irrelevance | 7 / 195 | 3.6% | 6.7% |
| ood_false_accept | tool_removed | 8 / 198 | 4.0% | 3.0% |
| permutation_flip | groups_disagreeing | 4 / 48 | 8.3% | 35.4% |
| semantic_confusion (count of errors) | tool:near_duplicate | 23 | | |
| tool_description (count of errors) | description_dependent | 19 | | |
| ambiguous (count of errors) | tool | 5 | | |

<details><summary>examples wrong:tool</summary>

```json
[
 {
  "context": "I am trying to log my workouts and calculate the total calories I've burned today. I weighed myself this morning, and I'm currently at 70kg. My activities inclu",
  "gold": "weightmanagement.activity_tracker",
  "pred": "calculate_calories_burned",
  "k": 4,
  "p_pred": 0.676,
  "language": null
 },
 {
  "context": "I have gathered a set of customer reviews and I need to understand the overall sentiment. Can you analyze it using a sentiment analysis algorithm for me?",
  "gold": "applyDataMiningAlgorithm",
  "pred": "runAlgorithm",
  "k": 8,
  "p_pred": 0.893,
  "language": null
 },
 {
  "context": "Who were the top new artists last year?",
  "gold": "Year End Top New Artists",
  "pred": "Top New Artists",
  "k": 16,
  "p_pred": 0.537,
  "language": null
 },
 {
  "context": "I need a list of restaurants near Av. Corrientes 348, Buenos Aires, Argentina.",
  "gold": "Get Restaurants by Address",
  "pred": "restaurants_v2_list",
  "k": 32,
  "p_pred": 0.445,
  "language": null
 },
 {
  "context": "Can you retrieve the height of the building with ID 101 on the 'buildings' layer? Also, I need the type of the road with ID 45 on the 'roads' layer.",
  "gold": "getLayerAttribute",
  "pred": "getBuildingDimensions",
  "k": 8,
  "p_pred": 0.442,
  "language": null
 },
 {
  "context": "Could you give me the current standings and points for NBA teams this season?",
  "gold": "Basketballv3Rankings",
  "pred": "Basketball Team Standings Seasons",
  "k": 16,
  "p_pred": 0.697,
  "language": null
 },
 {
  "context": "Could you please help me find the latitude and longitude of two places, one is the Eiffel Tower in Paris and the other is the Statue of Liberty in New York?",
  "gold": "navitime_geocoding",
  "pred": "reversegeocoding",
  "k": 32,
  "p_pred": 0.373,
  "language": null
 },
 {
  "context": "Could you find the latest news articles related to major developments in artificial intelligence in Indonesia for me?",
  "gold": "vivanews-search",
  "pred": "GetNewsByTopic",
  "k": 16,
  "p_pred": 0.774,
  "language": null
 }
]
```
</details>

<details><summary>examples ood:tool_removed</summary>

```json
[
 {
  "context": "I need to compare two branches named 'feature' and 'development' from the repository at 'https://github.com/user/project'. Can you do that for me?",
  "accepted": "gitRepoDataFetcher",
  "conf": 0.739,
  "options": [
   "team_streaks",
   "Search Books",
   "options_prices",
   "ethnic_food_stalls",
   "Flight Prices for a Week",
   "Flight Integration API"
  ]
 },
 {
  "context": "Could you find the latest news articles related to major developments in artificial intelligence in Indonesia for me?",
  "accepted": "Get Articles",
  "conf": 0.93,
  "options": [
   "Get Articles",
   "podcast.distribution_analytics",
   "video_post_metadata",
   "Get All Categories",
   "GetPeopleByTopic",
   "Get All Videos of Actor/Actress"
  ]
 },
 {
  "context": "I need to control the movement of my robot 'robot123' to move forward for 5 seconds. However, before that, can you check its current status?",
  "accepted": "controlRobotMovement",
  "conf": 0.97,
  "options": [
   "get_car_models",
   "GetNewsByTopic",
   "Get User Tweets",
   "PolicyImpactAnalysis.retrieveImpact",
   "EmailVerifications",
   "getpeoplebytopic"
  ]
 },
 {
  "context": "Role definition: Inquirer: A user who raises an inquiry. Response assistant: Communicates with the inquirer and provides answers and solutions. Historical dialo",
  "accepted": "Skyscanner Flights Results",
  "conf": 0.89,
  "options": [
   "Get Venue Details",
   "Convert Number To Speech",
   "products_v2_list_recommended",
   "Country Data API",
   "us_az",
   "Skyscanner Flights Results"
  ]
 }
]
```
</details>

<details><summary>examples ood:native_irrelevance</summary>

```json
[
 {
  "context": "Find the latitude and longitude coordinates for IP addresses '8.8.8.8', '1.1.1.1', and '74.125.24.100' using the ip-api.com API.",
  "accepted": "is_valid_ip_address",
  "conf": 0.831,
  "options": [
   "get_city_from_zipcode",
   "E-Sports Categories",
   "is_valid_ip_address",
   "project_population",
   "get_dna_sequence",
   "is_valid_email"
  ]
 },
 {
  "context": "Fetch the order details for ID '67890' and list all categories from AliExpress for the German locale.",
  "accepted": "listoforders",
  "conf": 0.825,
  "options": [
   "Get News from Sri Lanka",
   "list_of_room_facility_types",
   "Get Block Timestamp",
   "listoforders",
   "posts",
   "Greatest of All Time Mainstream Rock Songs"
  ]
 },
 {
  "context": "I'm planning a trip to Europe and Asia. Can you list the webcams available in Europe and Asia in English, showing images and locations?",
  "accepted": "webcams_list_country_country_country",
  "conf": 0.992,
  "options": [
   "influencer_campaign_create",
   "users.password.reset",
   "Warzone User Info",
   "webcams_list_country_country_country",
   "Get Contract Info",
   "getWindDirection"
  ]
 },
 {
  "context": "Search for images of classic literature book covers and also find images of modern science fiction book covers.",
  "accepted": "images_search",
  "conf": 0.953,
  "options": [
   "Retrieve Blur Events",
   "images_search",
   "Tours",
   "Abandoned Baby Indicator",
   "Canadian Gas Prices Locations",
   "deployApplication"
  ]
 }
]
```
</details>

