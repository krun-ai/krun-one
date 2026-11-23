"""Rule tables for technology-selection decisions (v1.1 R1/R2 corpus and KrunTechBench).

One table = one kind of thing (data store, messaging system, ...). Each option has, per property, Y (clearly yes),
N (clearly no) or P (partial / debatable), an overhead level (0 minimal, 1 moderate, 2 high) and optionally
``bench_only`` (never shown in training: the "unseen technology" slice).

How a decision is derived (tech.py): a requirement set R and a candidate subset S are sampled independently. Options
with P on any property of R are removed from S, so no label depends on a debatable cell. Gold = the unique option of S
that has Y on every property of R with the lowest overhead; ties are discarded; no satisfier = none-of-the-above.

Cells were written conservatively and then reviewed cell by cell by an independent pass (588 cells; findings in
results/v11/techbench/table_review.json). Every cell the review disputed (2 wrong, 26 debatable) is now P, the 23
properties whose two phrasings were not equivalent were reworded to the stricter reading, and one overhead level was
changed (canary release 2 -> 1). The exception is structure/insert, reworded to insertion only instead of downgraded.
The review was done by a model, not by a person: a human audit is still owed before these tables are called final.
"""

from __future__ import annotations

OVERHEAD = {0: "minimal", 1: "moderate", 2: "high"}


def _t(track, noun, props, options, fillers, max_req=3):
    keys = list(props)
    opts = {}
    for name, spec in options.items():
        ov, cells = spec[0], spec[1].split()
        assert len(cells) == len(keys), (name, len(cells), len(keys))
        assert all(c in "YNP" for c in cells), name
        opts[name] = {"overhead": ov, "props": dict(zip(keys, cells)), "bench_only": len(spec) > 2 and spec[2] == "bench"}
    return {"track": track, "noun": noun, "props": props, "options": opts, "fillers": fillers, "max_req": max_req}


TABLES = {
    # ------------------------------------------------------------------------------------------------ database / storage
    "datastore": _t("database", "data store", {
        "joins": (["ad-hoc sql joins across many tables", "reports that join several tables with sql"], "sql joins"),
        "acid": (["multi-row acid transactions", "transactions with rollback and isolation that span several rows"], "acid transactions"),
        "network": (["many application servers connecting to it over the network",
                     "concurrent access from dozens of separate backend hosts"], "network clients"),
        "horiz": (["write throughput that keeps growing by adding nodes", "horizontal scaling of writes across machines"],
                  "horizontal write scaling"),
        "inmem": (["microsecond reads from a store that keeps all of its data in memory", "in-memory access with microsecond latency"], "in-memory speed"),
        "fulltext": (["relevance-ranked full-text search", "keyword search over text ranked by relevance"], "full-text search"),
        "graph": (["multi-hop relationship traversals such as friends of friends as first-class queries",
                   "a native graph model where multi-hop traversals are first-class queries"], "graph traversal"),
        "columnar": (["columnar storage for aggregations over billions of rows", "columnar analytics over very large fact tables"],
                     "columnar analytics"),
        "tsdb": (["built-in retention policies and downsampling for time-stamped metrics",
                  "a store designed for metric series with automatic rollups and expiry"], "time-series retention"),
    }, {
        #                    joins acid net horiz inmem ftext graph col tsdb
        "sqlite":        (0, "Y Y N N P P P N N"),
        "postgresql":    (1, "Y Y Y P N P P N P"),
        "cockroachdb":   (2, "Y Y Y Y N P P N N"),
        "dynamodb":      (0, "N P Y Y P N N N N"),
        "cassandra":     (2, "N P Y Y N N N N P"),
        "redis":         (1, "N P Y P Y P N N P"),
        "elasticsearch": (2, "N N Y Y N Y N P P"),
        "clickhouse":    (1, "P N Y P N P N Y P"),
        "neo4j":         (1, "N Y Y P N P Y N N"),
        "influxdb":      (1, "P N Y P N N N P Y"),
        "mysql":         (1, "Y Y Y P N P P N N", "bench"),
        "memcached":     (1, "N N Y P Y N N N N", "bench"),
    }, ["security", "scaling", "deploystrategy"]),
    "storagekind": _t("database", "storage type", {
        "objects": (["billions of immutable files reachable over http with lifecycle rules",
                     "cheap durable storage for uploaded files addressed by key over an http api"], "http object storage"),
        "posix": (["a posix file system mounted by many machines at the same time",
                   "a shared directory that several servers read and write concurrently"], "shared posix file system"),
        "dbdisk": (["low-latency random reads and writes for one database server, surviving replacement of that server",
                    "a network-attached persistent disk for one database machine that outlives that machine"], "persistent block device"),
        "archive": (["the lowest price for data read less than once a year, where retrieval may take hours",
                     "the cheapest tier for backups that are almost never read and may take hours to restore"], "cold archive pricing"),
        "scratch": (["the highest iops for temporary data that may be lost when the machine stops",
                     "very fast scratch space with no durability requirement"], "ephemeral high iops"),
    }, {
        #                          objects posix dbdisk archive scratch
        "object storage":       (0, "Y P N P N"),
        "network file system":  (1, "N Y P N N"),
        "block volume":         (1, "N N Y N P"),
        "archive storage tier": (0, "P N N Y N"),
        "local nvme disk":      (1, "N N P N Y"),
    }, ["security", "apistyle"], max_req=1),
    # ------------------------------------------------------------------------------------------------ distributed systems
    "messaging": _t("distributed", "messaging system", {
        "durable": (["messages that survive a broker crash until they are consumed or expire",
                     "no message loss if the broker crashes before delivery"], "durable messages"),
        "replay": (["consumers that can re-read messages from days ago", "replaying several days of past events for a new consumer"],
                   "replayable log"),
        "routing": (["per-message routing rules by topic pattern and headers",
                     "routing each message to different consumers by rules on its topic and headers"], "flexible routing"),
        "ordered": (["strict per-key ordering at very high throughput", "ordered partitions handling millions of events per second"],
                    "high-throughput ordered partitions"),
        "iot": (["a lightweight protocol for battery-powered devices on unreliable networks",
                 "tiny sensors that publish over flaky mobile connections"], "lightweight iot protocol"),
    }, {
        #                   durable replay routing ordered iot
        "kafka":         (2, "Y Y N Y N"),
        "rabbitmq":      (1, "Y P Y P P"),
        "amazon sqs":    (0, "Y N N P N"),
        "nats core":     (1, "N N P P P"),
        "redis pub sub": (1, "N N P N N"),
        "mqtt broker":   (1, "P N P N Y"),
        "amazon kinesis": (0, "Y P N Y N", "bench"),
    }, ["security", "structure", "storagekind"], max_req=2),
    "replication": _t("distributed", "replication scheme", {
        "noloss": (["no acknowledged write lost when the primary node fails",
                    "every confirmed write surviving the loss of the leader"], "no acknowledged write lost on failover"),
        "multiwrite": (["accepting writes in several regions at the same time",
                        "clients writing to whichever region is closest"], "writes accepted in many regions"),
        "election": (["automatic leader election with one agreed order of writes",
                      "nodes agreeing on a single log and choosing a new leader by themselves"], "automatic leader election"),
        "readcopies": (["spreading read traffic over several copies", "serving read queries from several copies of the data"],
                       "read scaling on replicas"),
    }, {
        #                                             noloss multiwrite election readcopies
        "single leader with asynchronous replicas": (1, "N N P Y"),
        "single leader with a synchronous replica": (1, "Y N P P"),
        "multi leader replication":                 (2, "P Y N Y"),
        "raft consensus group":                     (2, "Y N Y P"),
        "single node with nightly backups":         (0, "N N N N"),
    }, ["security", "structure"], max_req=2),
    # ------------------------------------------------------------------------------------------------ backend
    "structure": _t("backend", "data structure", {
        "lookup": (["expected constant-time lookup by exact key", "o(1) average access by exact key"], "o(1) exact lookup"),
        "range": (["in-order iteration and range queries", "listing all keys between two bounds in sorted order"], "ordered range queries"),
        "insert": (["adding new elements constantly without rebuilding the structure", "cheap insertion as new items keep arriving"],
                   "fast inserts"),
        "minpop": (["repeatedly removing the smallest element", "always extracting the minimum element next"], "extract minimum"),
        "prefix": (["autocomplete by prefix", "finding every key that starts with a given prefix"], "prefix search"),
        "approx": (["a membership test in a few bits per item where rare false positives are fine",
                    "checking whether an item was probably seen, using almost no memory"], "approximate membership"),
        "evict": (["bounded memory with automatic eviction of the least recently used entries",
                   "a fixed-size store that drops whatever was used longest ago"], "lru eviction"),
    }, {
        #                          lookup range insert minpop prefix approx evict
        "hash table":           (0, "Y N Y N N N N"),
        "sorted array":         (0, "N Y N P P N N"),
        "binary heap":          (0, "N N Y Y N N N"),
        "balanced search tree": (1, "N Y Y Y P N N"),
        "trie":                 (1, "P P Y N Y N N"),
        "bloom filter":         (0, "N N Y N N Y N"),
        "lru cache":            (1, "Y N Y N N N Y"),
        "fifo queue":           (0, "N N Y N N N N"),
        "skip list":            (1, "N Y Y Y P N N", "bench"),
    }, ["security", "messaging", "deploystrategy"]),
    "scaling": _t("backend", "scaling pattern", {
        "repeat": (["the same expensive queries repeated constantly, where results a minute old are acceptable",
                    "identical read requests recomputed over and over although slightly stale answers are fine"], "reuses repeated results"),
        "static": (["static assets that load slowly for users on other continents",
                    "images and scripts served too slowly to distant visitors"], "serves static files near users"),
        "writes": (["write volume beyond what a single primary database can accept",
                    "more inserts per second than one database primary can handle"], "splits writes across databases"),
        "burst": (["bursts of background jobs that overwhelm workers, although the work may finish minutes later",
                   "spiky asynchronous work that does not need an immediate result"], "buffers work for later"),
        "abuse": (["a few clients flooding the api and starving everyone else",
                   "individual callers sending far more requests than their fair share"], "caps requests per client"),
        "cascade": (["a slow downstream dependency that makes callers hang until failures cascade",
                     "calls to a failing service piling up and taking the caller down with it"], "stops calling a failing dependency"),
        "cpu": (["stateless web servers that hit cpu limits at peak hours and sit idle at night",
                 "request load on stateless servers that swings widely during the day"], "adds and removes servers with load"),
        "fresh": (["arbitrary read queries that overload the primary although results must be nearly real time",
                   "heavy varied read traffic that needs fresh data"], "extra database copies for reads"),
    }, {
        #                       repeat static writes burst abuse cascade cpu fresh
        "cache with ttl":    (1, "Y P N N N N N P"),
        "cdn":               (1, "P Y N N N N N N"),
        "database sharding": (1, "N N Y N N N N P"),
        "message queue":     (1, "N N P Y N P N N"),
        "rate limiting":     (1, "N N N P Y N N N"),
        "circuit breaker":   (1, "N N N N N Y N N"),
        "autoscaling":       (1, "N N N P N N Y N"),
        "read replicas":     (1, "P N N N N N N Y"),
    }, ["security", "structure", "authn"], max_req=1),
    # ------------------------------------------------------------------------------------------------ cloud / infra
    "compute": _t("cloud", "compute platform", {
        "code": (["running server-side application code", "executing backend logic on each request"], "runs server code"),
        "longrun": (["processes that run for hours and hold long-lived connections",
                     "a daemon that stays up and keeps thousands of sockets open"], "long-running processes"),
        "zero": (["paying nothing for compute while there is no traffic", "no compute charges at all when idle"], "no cost when idle"),
        "orchestrate": (["dozens of services with service discovery, rolling updates and autoscaling managed together",
                         "one control plane that schedules and heals many containerized services"], "multi-service orchestration"),
        "kernel": (["full control of the operating system and kernel settings", "installing kernel modules and tuning sysctl"],
                   "full operating system control"),
    }, {
        #                              code longrun zero orchestrate kernel
        "static site on a cdn":      (0, "N N P N N"),
        "serverless functions":      (0, "Y N Y N N"),
        "managed container service": (1, "Y P P P P"),
        "single virtual machine":    (1, "Y Y N N Y"),
        "kubernetes cluster":        (2, "Y Y P Y P"),
        "bare metal server":         (2, "Y Y N N Y", "bench"),
    }, ["security", "structure", "messaging"]),
    # ------------------------------------------------------------------------------------------------ API / networking
    "apistyle": _t("api", "api style", {
        "browser": (["being called directly from browser javascript with no extra proxy",
                     "working from a web page with only what the browser ships"], "works from browsers"),
        "bidi": (["both sides sending messages at any time over one open connection",
                  "full two-way messaging on a single connection"], "two-way messaging"),
        "push": (["the server pushing a continuous stream of updates to a waiting client",
                  "live updates streamed to a connected client without it asking again"], "server push stream"),
        "binary": (["a strict binary contract with generated client and server code",
                    "compact binary messages defined in an interface file that generates stubs"], "generated binary contract"),
        "fields": (["clients choosing exactly which fields of related resources to fetch in one round trip",
                    "one request that returns a caller-defined selection of nested data"], "client-selected fields"),
        "cache": (["responses cached by ordinary http caches and cdns by url",
                   "letting intermediaries cache get responses"], "http cacheable"),
        "callback": (["notifying a partner's server when an event happens, with no polling and no connection held open",
                      "calling an url supplied by the customer whenever something changes"], "event callbacks to other servers"),
    }, {
        #                          browser bidi push binary fields cache callback
        "rest over http":       (0, "Y N N P P Y N"),
        "server sent events":   (0, "Y N Y N N N N"),
        "webhooks":             (0, "N N P N N N Y"),
        "websocket":            (1, "Y Y Y N N N N"),
        "graphql":              (1, "Y P P P Y P N"),
        "grpc":                 (1, "N Y Y Y P N N"),
    }, ["security", "structure", "storagekind"]),
    "netcomponent": _t("api", "network component", {
        "path": (["routing requests by url path or host header", "sending /api and /static to different backends"], "http path routing"),
        "tcp": (["balancing arbitrary tcp or udp traffic, not only http", "spreading raw tcp or udp connections across servers"],
                "tcp and udp balancing"),
        "quota": (["per-consumer api keys, quotas and request validation at the edge",
                   "metering each api customer and rejecting calls over their plan"], "api keys and quotas"),
        "edge": (["serving cached content from locations near each user", "caching responses at points of presence worldwide"],
                 "edge caching"),
        "private": (["encrypted private connectivity between two networks across the internet",
                     "joining an office network to a cloud network through an encrypted link"], "encrypted site-to-site link"),
        "mesh": (["automatic mutual tls and retries between internal services without changing application code",
                  "encrypting and retrying every call between internal services transparently"], "transparent service-to-service tls"),
    }, {
        #                            path tcp quota edge private mesh
        "layer 4 load balancer":  (1, "N Y N N N N"),
        "layer 7 load balancer":  (1, "Y N P N N N"),
        "api gateway":            (1, "Y N Y N N N"),
        "cdn":                    (1, "P N N Y N N"),
        "vpn tunnel":             (1, "N N N N Y N"),
        "service mesh":           (2, "Y P N N N Y"),
        "dns round robin":        (0, "N P N N N N"),
    }, ["structure", "storagekind", "deploystrategy"], max_req=2),
    # ------------------------------------------------------------------------------------------------ security
    "security": _t("security", "security mechanism", {
        "pw": (["storing user passwords so that a leaked database resists offline guessing",
                "a slow salted way to keep login passwords"], "slow salted password hashing"),
        "fingerprint": (["a fast fixed-size fingerprint of large files to spot duplicates, with no secret involved",
                         "a quick checksum-like identifier for file contents"], "fast content fingerprint"),
        "mac": (["two services that share a secret key checking that a message was not modified",
                 "tamper detection on webhooks using a shared secret"], "shared-key message authentication"),
        "pubverify": (["anyone verifying who published a release using only a public key",
                       "proof of origin that others can check but cannot forge"], "public-key signatures"),
        "atrest": (["stored records that are unreadable without a key but that the key holder can read back",
                    "reversible protection of data at rest"], "encryption of stored data"),
        "transit": (["protecting traffic between browser and server from eavesdropping",
                     "an encrypted channel for a network connection"], "encrypted transport"),
        "textsafe": (["representing binary bytes as plain text inside a json field, with no secrecy needed",
                      "a text-safe representation of raw bytes"], "binary-to-text encoding"),
    }, {
        #                                pw fingerprint mac pubverify atrest transit textsafe
        "argon2id password hashing": (1, "Y N N N N N N"),
        "sha 256 digest":            (0, "N Y N N N N N"),
        "hmac sha256":               (0, "N P Y N N N N"),
        "ed25519 signature":         (1, "N N P Y N N N"),
        "aes 256 gcm encryption":    (1, "N N P N Y P N"),
        "tls":                       (1, "N N N N N Y N"),
        "base64 encoding":           (0, "N N N N N N Y"),
        "bcrypt password hashing":   (1, "Y N N N N N N", "bench"),
    }, ["structure", "messaging", "storagekind"], max_req=1),
    "authn": _t("security", "authentication method", {
        "idp": (["users signing in through an external identity provider without giving the app their password",
                 "delegated login where the app only receives a token from the provider"], "delegated user login"),
        "nouser": (["service-to-service calls with no user present, using short-lived tokens from an authorization server",
                    "a backend job obtaining an expiring access token with its own credentials"], "machine-to-machine tokens"),
        "certs": (["both ends proving their identity with certificates at the transport layer",
                   "client certificates checked during the connection handshake"], "certificate authentication of both ends"),
        "phish": (["phishing-resistant sign-in with no password to type", "login with a public key credential instead of a password, resistant to phishing"],
                  "phishing-resistant passwordless login"),
        "revoke": (["ending a logged-in browser session immediately on the server side",
                    "invalidating a user's session at once from the backend"], "immediate server-side revocation"),
    }, {
        #                                             idp nouser certs phish revoke
        "oauth2 authorization code with pkce":      (1, "Y N N P P"),
        "oauth2 client credentials":                (1, "N Y N N P"),
        "mutual tls":                               (2, "N P Y P P"),
        "passkeys":                                 (1, "N N N Y P"),
        "server-side session cookie":               (0, "N N N N Y"),
        "static api key":                           (0, "N N N N P"),
    }, ["structure", "storagekind", "scaling"], max_req=1),
    # ------------------------------------------------------------------------------------------------ DevOps
    "deploystrategy": _t("devops", "release strategy", {
        "nodown": (["no downtime during the release", "users never seeing an outage while the new version goes out"], "no downtime"),
        "percent": (["exposing the new version to a small percentage of real users first",
                     "limiting the first exposure to a few percent of traffic"], "gradual exposure to real users"),
        "mirror": (["exercising the new version with real production requests while users never see its responses",
                    "comparing the new version against live traffic without affecting any user"], "tests on mirrored traffic"),
        "single": (["never having two versions serving at the same time",
                    "old and new code that must never both serve users"], "one version at a time"),
        "peruser": (["turning a capability on for specific customers without a new deployment",
                     "enabling behaviour per account at run time"], "per-customer toggles"),
        "spare": (["no second copy of the production environment", "no duplicate fleet to pay for"], "no duplicate environment"),
    }, {
        #                            nodown percent mirror single peruser spare
        "recreate deployment":    (0, "N N N Y N Y"),
        "rolling update":         (0, "Y P N N N P"),
        "blue green deployment":  (1, "Y P N P N N"),
        "feature flags":          (1, "P Y P N Y Y"),
        "canary release":         (1, "Y Y N N N P"),
        "traffic shadowing":      (2, "P N Y P N P"),
    }, ["security", "structure", "storagekind"], max_req=2),
    # ------------------------------------------------------------------------------------------------ ML infrastructure
    "mlserving": _t("mlinfra", "model serving mode", {
        "fast": (["answers within tens of milliseconds for requests that arrive one at a time",
                  "an interactive feature that needs each prediction within tens of milliseconds"], "low-latency single predictions"),
        "open": (["inputs that cannot be listed in advance", "arbitrary new inputs at request time"], "handles unseen inputs"),
        "offline": (["working on the device with no network connection", "predictions on a phone in airplane mode"],
                    "runs without a network"),
        "bulk": (["scoring hundreds of millions of rows overnight at the lowest cost",
                  "a nightly job that labels the whole table as cheaply as possible"], "cheap bulk scoring"),
        "events": (["scoring each event as it arrives on a message stream and writing the result downstream",
                    "continuous predictions inside a stream-processing pipeline"], "scores events on a stream"),
    }, {
        #                                     fast open offline bulk events
        "precomputed lookup table":         (0, "Y N P P N"),
        "offline batch inference":          (0, "N P N Y N"),
        "real-time online endpoint":        (1, "Y Y N N P"),
        "streaming inference":              (2, "P Y N N Y"),
        "on-device inference":              (2, "Y Y Y N N"),
    }, ["security", "structure", "storagekind"], max_req=2),
    "vectorindex": _t("mlinfra", "vector index", {
        "exact": (["exact nearest neighbours with perfect recall", "results guaranteed identical to a full scan"], "exact results"),
        "sublinear": (["query time that stays low at hundreds of millions of vectors",
                       "search time that grows much slower than the number of vectors"], "sub-linear search"),
        "compress": (["compressing vectors so the index fits in a fraction of the raw memory",
                      "quantized codes instead of full vectors in memory"], "compressed vectors"),
        "online": (["adding vectors continuously with no training step", "inserting new items one by one with no training or rebuild"],
                   "incremental inserts without training"),
    }, {
        #                 exact sublinear compress online
        "flat index":   (0, "Y N N Y"),
        "hnsw index":   (1, "N Y P Y"),
        "ivf pq index": (2, "N Y Y P"),
    }, ["security", "structure", "storagekind"], max_req=2),
}

INTROS = ["A team is building the backend of a food delivery app.", "A startup is designing an internal analytics tool.",
          "An engineer is reworking part of a payments platform.", "A logistics company is modernizing its tracking system.",
          "A small team maintains a multiplayer game backend.", "A hospital software group is planning a new service.",
          "A developer is adding a feature to an e-commerce site.", "A fintech team is reviewing a design proposal.",
          "A media company is rebuilding its content pipeline.", "An open-source project is choosing its building blocks.",
          "A telecom operator is replacing a legacy component.", "A research lab is setting up shared infrastructure.",
          "A travel booking site is splitting up its monolith.", "An insurance company is automating claims handling.",
          "A robotics startup is building its fleet console.", "A university is replacing its course registration system.",
          "A marketplace is preparing for a seasonal traffic peak.", "A bank is opening an api for partners.",
          "A streaming service is reworking its recommendations.", "A government agency is digitizing a paper process."]
